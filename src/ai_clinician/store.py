"""Append-only aggregate research store with a tamper-evident audit chain."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .feature_manifest import AnalysisManifestRegistration
from .identity import GovernanceDirectory
from .models import (
    ALLOWED_LIFECYCLE_TRANSITIONS,
    CandidateDiseaseState,
    CausalAuditArtifact,
    CohortSnapshot,
    DataProfileRegistration,
    ExpertReview,
    GuidelineChannel,
    GuidelineRecommendation,
    GuidelineRelease,
    LifecycleStatus,
    MVP_DISABLED_LIFECYCLE_STATES,
    ModelRelease,
    ReleaseStatus,
    ReviewDecision,
    ReviewRole,
    ResearchStudy,
    StatePartitionSnapshot,
    StateValidationReport,
    TargetTrialInputSnapshot,
    TargetTrialResult,
    TargetTrialSpec,
    TimelineQualityReport,
    TimelineValidationManifestRegistration,
    transition_study,
)
from .privacy import (
    PrivacyBoundaryError,
    PrivacyConfig,
    assert_aggregate_safe_payload,
    is_placeholder_secret,
)


_OBJECT_SCHEMAS: dict[str, type[BaseModel]] = {
    "study": ResearchStudy,
    "data_profile": DataProfileRegistration,
    "cohort_snapshot": CohortSnapshot,
    "timeline_validation_manifest": TimelineValidationManifestRegistration,
    "timeline_quality_report": TimelineQualityReport,
    "analysis_manifest_registration": AnalysisManifestRegistration,
    "state_partition_snapshot": StatePartitionSnapshot,
    "candidate_state": CandidateDiseaseState,
    "state_validation_report": StateValidationReport,
    "model_release": ModelRelease,
    "target_trial_spec": TargetTrialSpec,
    "target_trial_input_snapshot": TargetTrialInputSnapshot,
    "guideline_recommendation": GuidelineRecommendation,
    "guideline_release": GuidelineRelease,
    "expert_review": ExpertReview,
}

_FHIR_CPG_RESOURCE_TYPES = {"Library", "ActivityDefinition", "PlanDefinition"}
_DISABLED_OBJECT_TYPES = {"causal_audit_artifact", "target_trial_result"}
_STUDY_BINDING_FIELDS = (
    "data_profile_id",
    "timeline_quality_report_id",
    "cohort_snapshot_id",
    "state_model_release_id",
    "target_trial_ids",
    "guideline_release_id",
    "guideline_release_version",
    "guideline_release_hash",
)
_STUDY_ALLOWED_BINDINGS: dict[LifecycleStatus, frozenset[str]] = {
    LifecycleStatus.DRAFT: frozenset(),
    LifecycleStatus.DATA_REGISTERED: frozenset({"data_profile_id"}),
    LifecycleStatus.TIMELINE_VALIDATED: frozenset(
        {"data_profile_id", "timeline_quality_report_id"}
    ),
    LifecycleStatus.COHORT_LOCKED: frozenset(
        {"data_profile_id", "timeline_quality_report_id", "cohort_snapshot_id"}
    ),
    LifecycleStatus.STATES_DISCOVERED: frozenset(
        {
            "data_profile_id",
            "timeline_quality_report_id",
            "cohort_snapshot_id",
            "state_model_release_id",
        }
    ),
    LifecycleStatus.TARGET_TRIALS_COMPLETED: frozenset(
        {
            "data_profile_id",
            "timeline_quality_report_id",
            "cohort_snapshot_id",
            "state_model_release_id",
            "target_trial_ids",
        }
    ),
    LifecycleStatus.EXPERT_REVIEW: frozenset(
        {
            "data_profile_id",
            "timeline_quality_report_id",
            "cohort_snapshot_id",
            "state_model_release_id",
            "target_trial_ids",
        }
    ),
    LifecycleStatus.CANDIDATE_GUIDELINE: frozenset(_STUDY_BINDING_FIELDS),
    LifecycleStatus.RESEARCH_RELEASE: frozenset(_STUDY_BINDING_FIELDS),
    LifecycleStatus.RETIRED: frozenset(_STUDY_BINDING_FIELDS),
}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ResearchStore:
    """Stores only versioned aggregate artefacts; source clinical text is refused."""

    def __init__(
        self,
        relative_database_path: str | Path,
        *,
        privacy: PrivacyConfig,
        audit_signing_key: str | bytes | None = None,
        governance_directory: GovernanceDirectory | None = None,
    ) -> None:
        self.path = privacy.resolve_derived(relative_database_path, create_parent=True)
        configured_key = audit_signing_key or os.environ.get(
            "AI_CLINICIAN_AUDIT_SIGNING_KEY"
        )
        if isinstance(configured_key, str):
            configured_key = configured_key.encode("utf-8")
        if configured_key is not None and len(configured_key) < 32:
            raise ValueError("audit signing key must contain at least 32 bytes")
        if configured_key is not None and is_placeholder_secret(configured_key):
            raise ValueError("audit signing key cannot be a placeholder")
        self._audit_signing_key = configured_key
        self._governance_directory = governance_directory
        self._initialize()

    @staticmethod
    def _required_write_roles(
        object_type: str, payload: Mapping[str, Any]
    ) -> tuple[str, ...]:
        if object_type == "expert_review":
            return (str(payload.get("role", "")),)
        if object_type == "guideline_release":
            if payload.get("channel") == "baseline_approved" and payload.get(
                "status"
            ) in {"locked", "released"}:
                return ("expert_committee",)
            return ("research_governance", "expert_committee")
        if object_type == "guideline_recommendation":
            return (
                ("expert_committee", "crc_clinical_expert")
                if payload.get("channel") == "baseline_approved"
                else (
                    "research_governance",
                    "crc_clinical_expert",
                    "methodologist",
                )
            )
        return {
            "study": ("research_governance", "data_steward"),
            "data_profile": ("data_steward",),
            "cohort_snapshot": ("data_steward",),
            "timeline_validation_manifest": ("data_steward",),
            "timeline_quality_report": ("data_steward", "methodologist"),
            "analysis_manifest_registration": ("data_steward",),
            "state_partition_snapshot": ("methodologist",),
            "candidate_state": ("methodologist", "research_governance"),
            "state_validation_report": ("methodologist",),
            "model_release": ("methodologist", "research_governance"),
            "target_trial_spec": ("methodologist",),
            "target_trial_input_snapshot": ("data_steward", "methodologist"),
        }.get(object_type, ())

    def _authenticate_write(
        self,
        object_type: str,
        payload: Mapping[str, Any],
        governance_token: str | None,
    ) -> str:
        roles = self._required_write_roles(object_type, payload)
        if not roles:
            return "deterministic-export-engine"
        directory = self._governance_directory or GovernanceDirectory.from_env()
        identity = directory.authenticate(governance_token, allowed_roles=roles)
        if object_type == "expert_review" and str(payload.get("role")) not in identity.roles:
            raise ValueError("review role is not held by the authenticated identity")
        return identity.subject

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_objects (
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version > 0),
                    payload_json TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (object_type, object_id, version)
                );
                CREATE TABLE IF NOT EXISTS audit_chain (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    object_version INTEGER NOT NULL,
                    actor TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL UNIQUE,
                    entry_signature TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS research_objects_no_update
                BEFORE UPDATE ON research_objects BEGIN
                    SELECT RAISE(ABORT, 'research objects are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS research_objects_no_delete
                BEFORE DELETE ON research_objects BEGIN
                    SELECT RAISE(ABORT, 'research objects are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS audit_chain_no_update
                BEFORE UPDATE ON audit_chain BEGIN
                    SELECT RAISE(ABORT, 'audit chain is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS audit_chain_no_delete
                BEFORE DELETE ON audit_chain BEGIN
                    SELECT RAISE(ABORT, 'audit chain is append-only');
                END;
                """
            )

    @staticmethod
    def _validate_fhir_cpg_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(value)
        entries = payload.get("entry")
        if (
            payload.get("resourceType") != "Bundle"
            or payload.get("type") != "collection"
            or not isinstance(entries, list)
            or not entries
        ):
            raise ValueError("FHIR CPG export must be a non-empty collection Bundle")
        resource_types: list[str] = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("resource"), Mapping
            ):
                raise ValueError("FHIR CPG Bundle entries require resources")
            resource_type = str(entry["resource"].get("resourceType", ""))
            if resource_type not in _FHIR_CPG_RESOURCE_TYPES:
                raise ValueError("FHIR CPG Bundle contains a forbidden resource type")
            resource_types.append(resource_type)
        if "Library" not in resource_types or "PlanDefinition" not in resource_types:
            raise ValueError("FHIR CPG Bundle requires Library and PlanDefinition resources")
        assert_aggregate_safe_payload(payload)
        return payload

    @classmethod
    def _payload(
        cls,
        object_type: str,
        object_id: str,
        value: BaseModel | Mapping[str, Any],
    ) -> dict[str, Any]:
        if object_type == "fhir_cpg_bundle":
            if isinstance(value, BaseModel) or not isinstance(value, Mapping):
                raise TypeError("FHIR CPG exports require the generated Bundle mapping")
            return cls._validate_fhir_cpg_bundle(value)
        expected = _OBJECT_SCHEMAS.get(object_type)
        if expected is None:
            raise ValueError(f"unsupported aggregate research object type: {object_type}")
        if type(value) is not expected:
            raise TypeError(
                f"{object_type} requires an exact {expected.__name__} instance; "
                "generic mappings are forbidden"
            )
        validated = expected.model_validate(value.model_dump(mode="json"))
        payload = validated.model_dump(mode="json")
        payload_id = payload.get("id")
        if payload_id is not None and payload_id != object_id:
            raise ValueError("research object id must match the validated payload id")
        assert_aggregate_safe_payload(payload)
        return payload

    @staticmethod
    def _stored_record(
        connection: sqlite3.Connection,
        object_type: str,
        object_id: str,
        *,
        version: int | None = None,
    ) -> sqlite3.Row | None:
        if version is None:
            return connection.execute(
                """SELECT * FROM research_objects
                   WHERE object_type = ? AND object_id = ?
                   ORDER BY version DESC LIMIT 1""",
                (object_type, object_id),
            ).fetchone()
        return connection.execute(
            """SELECT * FROM research_objects
               WHERE object_type = ? AND object_id = ? AND version = ?""",
            (object_type, object_id, version),
        ).fetchone()

    @staticmethod
    def _row_has_signature(
        connection: sqlite3.Connection, row: sqlite3.Row | None
    ) -> bool:
        if row is None:
            return False
        signature = connection.execute(
            """SELECT entry_signature FROM audit_chain
               WHERE object_type = ? AND object_id = ? AND object_version = ?""",
            (row["object_type"], row["object_id"], row["version"]),
        ).fetchone()
        return bool(signature and signature["entry_signature"])

    def _validate_cross_object_contract(
        self,
        connection: sqlite3.Connection,
        object_type: str,
        object_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if object_type == "study":
            study = ResearchStudy.model_validate(payload)
            if study.state in MVP_DISABLED_LIFECYCLE_STATES:
                raise ValueError(
                    f"study state {study.state.value} is disabled in the research MVP"
                )
            prior_row = self._stored_record(connection, "study", object_id)
            allowed_bindings = _STUDY_ALLOWED_BINDINGS[study.state]
            if any(
                field not in allowed_bindings and getattr(study, field)
                for field in _STUDY_BINDING_FIELDS
            ):
                raise ValueError("study contains evidence bindings from a future lifecycle state")
            if prior_row is None:
                if study.state is not LifecycleStatus.DRAFT:
                    raise ValueError("a new study must start in draft state")
                return

            prior = ResearchStudy.model_validate_json(prior_row["payload_json"])
            if study.created_at != prior.created_at:
                raise ValueError("study creation time is immutable")
            if study.updated_at < prior.updated_at:
                raise ValueError("study update time cannot move backwards")
            if study.state is prior.state:
                if any(
                    getattr(study, field) != getattr(prior, field)
                    for field in _STUDY_BINDING_FIELDS
                ):
                    raise ValueError(
                        "study evidence bindings can change only during a lifecycle transition"
                    )
                return
            if study.state not in ALLOWED_LIFECYCLE_TRANSITIONS[prior.state]:
                raise ValueError(
                    f"invalid research lifecycle transition: {prior.state} -> {study.state}"
                )
            if any(
                getattr(prior, field) and getattr(study, field) != getattr(prior, field)
                for field in _STUDY_BINDING_FIELDS
            ):
                raise ValueError("existing study evidence bindings are immutable")

            # Validate the model-level transition contract and then repeat the full
            # deterministic evidence gate at the persistence boundary. This makes
            # direct ResearchStore callers subject to the same rules as the workflow.
            transition_study(
                study.model_copy(update={"state": prior.state}), study.state
            )
            from .orchestrator import ResearchWorkflow, WorkflowGateError

            try:
                ResearchWorkflow(self)._validate_gate(study, study.state)
            except WorkflowGateError as exc:
                raise ValueError(f"study lifecycle gate failed: {exc}") from exc
            return

        if object_type == "fhir_cpg_bundle":
            recommendation_row = self._stored_record(
                connection, "guideline_recommendation", object_id
            )
            if recommendation_row is None:
                raise ValueError("FHIR CPG Bundle requires its stored recommendation")
            recommendation = GuidelineRecommendation.model_validate_json(
                recommendation_row["payload_json"]
            )
            from .guidelines import export_cpg_on_fhir

            expected = export_cpg_on_fhir(recommendation).as_bundle()
            if not hmac.compare_digest(_canonical_json(payload), _canonical_json(expected)):
                raise ValueError("FHIR CPG Bundle differs from the deterministic export")
            return

        if object_type == "cohort_snapshot":
            cohort = CohortSnapshot.model_validate(payload)
            study_row = self._stored_record(connection, "study", cohort.study_id)
            profile_row = self._stored_record(
                connection, "data_profile", cohort.data_profile_id
            )
            if study_row is None or profile_row is None:
                raise ValueError("cohort snapshot requires its registered study and data profile")
            study = ResearchStudy.model_validate_json(study_row["payload_json"])
            profile = DataProfileRegistration.model_validate_json(
                profile_row["payload_json"]
            )
            if (
                study.data_profile_id != cohort.data_profile_id
                or study.state is not LifecycleStatus.TIMELINE_VALIDATED
                or profile.source_snapshot_sha256 != cohort.source_data_sha256
                or profile.aggregate_patient_count < cohort.eligible_patient_count
            ):
                raise ValueError("cohort snapshot provenance does not match the locked study")
            return

        if object_type == "guideline_release":
            release = GuidelineRelease.model_validate(payload)
            for recommendation_id, expected_hash in release.recommendation_hashes.items():
                rows = connection.execute(
                    """SELECT * FROM research_objects
                       WHERE object_type = 'guideline_recommendation' AND object_id = ?""",
                    (recommendation_id,),
                ).fetchall()
                if not any(
                    self._row_has_signature(connection, row)
                    and GuidelineRecommendation.model_validate_json(
                        row["payload_json"]
                    ).content_hash()
                    == expected_hash
                    for row in rows
                ):
                    raise ValueError(
                        "guideline release references an absent recommendation version"
                    )
            if release.status in {ReleaseStatus.LOCKED, ReleaseStatus.RELEASED}:
                from .guidelines import release_review_hash

                expected_review_hash = release_review_hash(release)
                reviews: list[ExpertReview] = []
                for review_id in release.expert_review_ids:
                    review_row = self._stored_record(
                        connection, "expert_review", review_id
                    )
                    signature_row = connection.execute(
                        """SELECT entry_signature FROM audit_chain
                           WHERE object_type = 'expert_review' AND object_id = ?
                           ORDER BY object_version DESC LIMIT 1""",
                        (review_id,),
                    ).fetchone()
                    if (
                        review_row is None
                        or signature_row is None
                        or not signature_row["entry_signature"]
                    ):
                        raise ValueError("locked guideline release lacks a signed review")
                    review = ExpertReview.model_validate_json(
                        review_row["payload_json"]
                    )
                    if (
                        review.object_id != release.id
                        or review.object_hash != expected_review_hash
                        or review.decision is not ReviewDecision.APPROVE
                        or not review.independent
                    ):
                        raise ValueError("locked guideline release has an invalid review")
                    reviews.append(review)
                minimum = 3 if release.channel is GuidelineChannel.BASELINE_APPROVED else 2
                if len({review.reviewer_key for review in reviews}) < minimum:
                    raise ValueError("locked guideline release reviews are not independent")
                roles = {review.role for review in reviews}
                if release.channel is GuidelineChannel.BASELINE_APPROVED:
                    clinical_count = sum(
                        review.role is ReviewRole.CRC_CLINICAL_EXPERT
                        for review in reviews
                    )
                    if clinical_count < 2 or not roles & {
                        ReviewRole.METHODOLOGIST,
                        ReviewRole.SAFETY_REVIEWER,
                    }:
                        raise ValueError(
                            "locked baseline release lacks its expert committee"
                        )
                elif not {
                    ReviewRole.CRC_CLINICAL_EXPERT,
                    ReviewRole.METHODOLOGIST,
                }.issubset(roles):
                    raise ValueError("locked candidate release lacks required review roles")
            return

        if object_type == "guideline_recommendation":
            recommendation = GuidelineRecommendation.model_validate(payload)
            if recommendation.status not in {
                ReleaseStatus.DRAFT,
                ReleaseStatus.IN_REVIEW,
            }:
                raise ValueError(
                    "recommendations cannot be locked or released outside a release package"
                )
            if not recommendation.research_only:
                raise ValueError("MVP recommendations must remain research-only")
            return

        if object_type == "expert_review":
            review = ExpertReview.model_validate(payload)
            candidates: list[tuple[str, type[BaseModel]]] = [
                ("guideline_recommendation", GuidelineRecommendation),
                ("guideline_release", GuidelineRelease),
                ("model_release", ModelRelease),
                ("candidate_state", CandidateDiseaseState),
                ("timeline_validation_manifest", TimelineValidationManifestRegistration),
                ("analysis_manifest_registration", AnalysisManifestRegistration),
            ]
            matches: list[tuple[str, BaseModel, str]] = []
            for candidate_type, model_type in candidates:
                row = self._stored_record(connection, candidate_type, review.object_id)
                if row is None or not self._row_has_signature(connection, row):
                    continue
                artifact = model_type.model_validate_json(row["payload_json"])
                if candidate_type == "guideline_release":
                    from .guidelines import release_review_hash

                    expected_hash = release_review_hash(artifact)
                else:
                    expected_hash = artifact.content_hash()  # type: ignore[attr-defined]
                matches.append((candidate_type, artifact, expected_hash))
            valid = [item for item in matches if item[2] == review.object_hash]
            if len(valid) != 1:
                raise ValueError("expert review must bind one current reviewable artifact")
            if valid[0][0] in {
                "candidate_state",
                "timeline_validation_manifest",
                "analysis_manifest_registration",
            } and review.role is not ReviewRole.CRC_CLINICAL_EXPERT:
                raise ValueError("artifact requires CRC clinical expert review")
            return

        if object_type == "target_trial_spec":
            spec = TargetTrialSpec.model_validate(payload)
            study_row = self._stored_record(connection, "study", spec.study_id)
            cohort_row = self._stored_record(
                connection, "cohort_snapshot", spec.cohort_snapshot_id
            )
            if study_row is None or cohort_row is None:
                raise ValueError("target-trial protocol requires a signed study and cohort")
            study_signature = connection.execute(
                """SELECT entry_signature FROM audit_chain
                   WHERE object_type = 'study' AND object_id = ?
                   ORDER BY object_version DESC LIMIT 1""",
                (spec.study_id,),
            ).fetchone()
            cohort_signature = connection.execute(
                """SELECT entry_signature FROM audit_chain
                   WHERE object_type = 'cohort_snapshot' AND object_id = ?
                   ORDER BY object_version DESC LIMIT 1""",
                (spec.cohort_snapshot_id,),
            ).fetchone()
            study = ResearchStudy.model_validate_json(study_row["payload_json"])
            cohort = CohortSnapshot.model_validate_json(cohort_row["payload_json"])
            if (
                study_signature is None
                or not study_signature["entry_signature"]
                or cohort_signature is None
                or not cohort_signature["entry_signature"]
                or cohort.content_hash() != spec.cohort_snapshot_hash
                or cohort.study_id != study.id
                or study.cohort_snapshot_id != cohort.id
                or spec.data_snapshot_id != cohort.id
                or study.state
                not in {
                    LifecycleStatus.COHORT_LOCKED,
                    LifecycleStatus.STATES_DISCOVERED,
                    LifecycleStatus.TARGET_TRIALS_COMPLETED,
                    LifecycleStatus.EXPERT_REVIEW,
                    LifecycleStatus.CANDIDATE_GUIDELINE,
                    LifecycleStatus.RESEARCH_RELEASE,
                }
            ):
                raise ValueError("target-trial study/cohort provenance is invalid")
            release_row = self._stored_record(
                connection,
                "guideline_release",
                spec.baseline_release_id,
                version=spec.baseline_release_version,
            )
            if release_row is None or release_row["content_sha256"] != spec.baseline_release_hash:
                raise ValueError("target-trial baseline release version/hash mismatch")
            if not self._row_has_signature(connection, release_row):
                raise ValueError("target-trial baseline release is not signature-bound")
            release = GuidelineRelease.model_validate_json(release_row["payload_json"])
            if (
                release.channel is not GuidelineChannel.BASELINE_APPROVED
                or release.status not in {ReleaseStatus.LOCKED, ReleaseStatus.RELEASED}
                or not set(spec.baseline_recommendation_ids).issubset(
                    release.recommendation_ids
                )
            ):
                raise ValueError("target-trial baseline is not an approved locked release")
            supported_strategies: set[str] = set()
            for recommendation_id in spec.baseline_recommendation_ids:
                expected_hash = spec.baseline_recommendation_hashes[recommendation_id]
                if release.recommendation_hashes.get(recommendation_id) != expected_hash:
                    raise ValueError("target-trial baseline recommendation hash mismatch")
                recommendation_rows = connection.execute(
                    """SELECT * FROM research_objects
                       WHERE object_type = 'guideline_recommendation' AND object_id = ?""",
                    (recommendation_id,),
                ).fetchall()
                matching = [
                    GuidelineRecommendation.model_validate_json(row["payload_json"])
                    for row in recommendation_rows
                    if self._row_has_signature(connection, row)
                    and GuidelineRecommendation.model_validate_json(
                        row["payload_json"]
                    ).content_hash()
                    == expected_hash
                ]
                if len(matching) != 1 or matching[0].channel is not GuidelineChannel.BASELINE_APPROVED:
                    raise ValueError("target-trial baseline recommendation is not immutable evidence")
                supported_strategies.update(
                    option.regimen_class
                    for option in matching[0].treatment_options
                    if option.guideline_backed
                    and option.decision_point is spec.decision_point
                )
            if not set(spec.strategies).issubset(supported_strategies):
                raise ValueError("target-trial strategy is unsupported by its approved baseline")
            return

        if object_type == "target_trial_result":
            result = TargetTrialResult.model_validate(payload)
            spec_row = self._stored_record(
                connection, "target_trial_spec", result.spec.id
            )
            input_row = self._stored_record(
                connection, "target_trial_input_snapshot", result.input_snapshot.id
            )
            if (
                spec_row is None
                or TargetTrialSpec.model_validate_json(spec_row["payload_json"])
                .content_hash()
                != result.spec.content_hash()
                or input_row is None
                or TargetTrialInputSnapshot.model_validate_json(
                    input_row["payload_json"]
                ).content_hash()
                != result.input_snapshot.content_hash()
            ):
                raise ValueError("target-trial result lacks its signed protocol/input evidence")
            for audit in [result.negative_control_audit, *result.sensitivity_audits]:
                audit_row = self._stored_record(
                    connection, "causal_audit_artifact", audit.id
                )
                if (
                    audit_row is None
                    or CausalAuditArtifact.model_validate_json(
                        audit_row["payload_json"]
                    ).content_hash()
                    != audit.content_hash()
                ):
                    raise ValueError("target-trial result lacks a matching causal audit")

    def put(
        self,
        object_type: str,
        object_id: str,
        value: BaseModel | Mapping[str, Any],
        *,
        actor: str = "system",
        governance_token: str | None = None,
    ) -> dict[str, Any]:
        """Append one immutable version and its audit entry atomically."""

        identifier_pattern = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
        if not identifier_pattern.fullmatch(object_type) or not identifier_pattern.fullmatch(
            object_id
        ):
            raise ValueError("object_type and object_id are required")
        if not identifier_pattern.fullmatch(actor):
            raise ValueError("audit actor must be a non-identifying governance subject key")
        if object_id.casefold().startswith("pt_"):
            raise PrivacyBoundaryError(
                "patient pseudonyms cannot be aggregate research object ids"
            )
        if object_type.casefold() in {
            "patient",
            "timeline",
            "patient_timeline",
            "clinical_event",
            "clinical_note",
            "raw_record",
        }:
            raise PrivacyBoundaryError(
                "patient-level object types are forbidden in the research store"
            )
        with self._connect() as audit_connection:
            existing_audit_entries = int(
                audit_connection.execute("SELECT COUNT(*) FROM audit_chain").fetchone()[0]
            )
        if existing_audit_entries and not self.verify_audit_chain():
            raise ValueError("audit chain verification failed before write")
        assert_aggregate_safe_payload(
            {"object_type": object_type, "object_id": object_id}
        )
        payload = self._payload(object_type, object_id, value)
        authenticated_actor = self._authenticate_write(
            object_type, payload, governance_token
        )
        if actor != "system" and actor != authenticated_actor:
            raise ValueError("caller-supplied actor differs from authenticated identity")
        actor = authenticated_actor
        if object_type == "expert_review" and payload.get("reviewer_key") != actor:
            raise ValueError("expert review actor must match its authenticated reviewer key")
        requires_signature = object_type in _OBJECT_SCHEMAS
        if requires_signature and self._audit_signing_key is None:
            raise ValueError("governed research artifacts require signed audit entries")
        encoded = _canonical_json(payload)
        content_hash = _sha256(encoded)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_cross_object_contract(
                connection, object_type, object_id, payload
            )
            row = connection.execute(
                """SELECT COALESCE(MAX(version), 0) AS latest
                   FROM research_objects WHERE object_type = ? AND object_id = ?""",
                (object_type, object_id),
            ).fetchone()
            version = int(row["latest"]) + 1
            if version > 1 and object_type in {
                "expert_review",
                "state_validation_report",
                "target_trial_result",
                "timeline_validation_manifest",
                "target_trial_spec",
                "causal_audit_artifact",
                "timeline_quality_report",
                "data_profile",
                "cohort_snapshot",
                "analysis_manifest_registration",
                "state_partition_snapshot",
                "target_trial_input_snapshot",
            }:
                raise ValueError(f"{object_type} ids are immutable; issue a new id")
            if version > 1 and object_type == "guideline_release":
                prior = connection.execute(
                    """SELECT payload_json FROM research_objects
                       WHERE object_type = ? AND object_id = ?
                       ORDER BY version DESC LIMIT 1""",
                    (object_type, object_id),
                ).fetchone()
                prior_status = json.loads(prior["payload_json"]).get("status")
                next_status = payload.get("status")
                allowed_transition = {
                    "in_review": "locked",
                    "locked": "released",
                }.get(prior_status)
                if next_status != allowed_transition:
                    raise ValueError(
                        "guideline release status can only advance in_review to locked to released"
                    )
            if version > 1 and object_type in {"candidate_state", "model_release"}:
                prior = connection.execute(
                    """SELECT payload_json FROM research_objects
                       WHERE object_type = ? AND object_id = ?
                       ORDER BY version DESC LIMIT 1""",
                    (object_type, object_id),
                ).fetchone()
                prior_payload = json.loads(prior["payload_json"])
                if prior_payload.get("eligible_for_guideline") or prior_payload.get(
                    "status"
                ) in {"locked", "released"}:
                    raise ValueError(
                        f"locked {object_type} is immutable; issue a new id"
                    )
            previous = connection.execute(
                "SELECT entry_hash FROM audit_chain ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = str(previous["entry_hash"]) if previous else "0" * 64
            audit_material = _canonical_json(
                {
                    "action": "append",
                    "actor": actor,
                    "created_at": created_at,
                    "object_id": object_id,
                    "object_type": object_type,
                    "object_version": version,
                    "payload_sha256": content_hash,
                    "previous_hash": previous_hash,
                }
            )
            entry_hash = _sha256(audit_material)
            entry_signature = (
                hmac.new(
                    self._audit_signing_key,
                    entry_hash.encode("ascii"),
                    hashlib.sha256,
                ).hexdigest()
                if self._audit_signing_key is not None
                else ""
            )
            connection.execute(
                """INSERT INTO research_objects
                   (object_type, object_id, version, payload_json, content_sha256, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (object_type, object_id, version, encoded, content_hash, created_at),
            )
            connection.execute(
                """INSERT INTO audit_chain
                   (action, object_type, object_id, object_version, actor, payload_sha256,
                    previous_hash, entry_hash, entry_signature, created_at)
                   VALUES ('append', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    object_type,
                    object_id,
                    version,
                    actor,
                    content_hash,
                    previous_hash,
                    entry_hash,
                    entry_signature,
                    created_at,
                ),
            )
        return {
            "object_type": object_type,
            "object_id": object_id,
            "version": version,
            "content_sha256": content_hash,
            "audit_hash": entry_hash,
        }

    def get(self, object_type: str, object_id: str, version: int | None = None) -> dict[str, Any] | None:
        if object_type in _DISABLED_OBJECT_TYPES:
            return None
        where = "object_type = ? AND object_id = ?"
        parameters: list[Any] = [object_type, object_id]
        if version is not None:
            where += " AND version = ?"
            parameters.append(version)
        order = "" if version is not None else " ORDER BY version DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM research_objects WHERE {where}{order}", parameters
            ).fetchone()
        if row is None:
            return None
        return {
            "object_type": row["object_type"],
            "object_id": row["object_id"],
            "version": row["version"],
            "content_sha256": row["content_sha256"],
            "created_at": row["created_at"],
            "payload": json.loads(row["payload_json"]),
        }

    def list_latest(self, object_type: str, *, limit: int = 100) -> list[dict[str, Any]]:
        if object_type in _DISABLED_OBJECT_TYPES:
            return []
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT item.* FROM research_objects AS item
                   JOIN (
                       SELECT object_id, MAX(version) AS version
                       FROM research_objects WHERE object_type = ? GROUP BY object_id
                   ) AS latest
                   ON item.object_id = latest.object_id AND item.version = latest.version
                   WHERE item.object_type = ?
                   ORDER BY item.created_at DESC LIMIT ?""",
                (object_type, object_type, limit),
            ).fetchall()
        return [
            {
                "object_type": row["object_type"],
                "object_id": row["object_id"],
                "version": row["version"],
                "content_sha256": row["content_sha256"],
                "created_at": row["created_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def list_all_latest(self, object_type: str) -> list[dict[str, Any]]:
        """Return every latest aggregate object of a type without a silent cap."""

        if object_type in _DISABLED_OBJECT_TYPES:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT item.* FROM research_objects AS item
                   JOIN (
                       SELECT object_id, MAX(version) AS version
                       FROM research_objects WHERE object_type = ? GROUP BY object_id
                   ) AS latest
                   ON item.object_id = latest.object_id AND item.version = latest.version
                   WHERE item.object_type = ? ORDER BY item.created_at""",
                (object_type, object_type),
            ).fetchall()
        return [
            {
                "object_type": row["object_type"],
                "object_id": row["object_id"],
                "version": row["version"],
                "content_sha256": row["content_sha256"],
                "created_at": row["created_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def artifact_has_valid_signature(self, object_type: str, object_id: str) -> bool:
        if object_type in _DISABLED_OBJECT_TYPES:
            return False
        if self._audit_signing_key is None or not self.verify_audit_chain():
            return False
        record = self.get(object_type, object_id)
        if record is None:
            return False
        with self._connect() as connection:
            row = connection.execute(
                """SELECT entry_signature FROM audit_chain
                   WHERE object_type = ? AND object_id = ? AND object_version = ?""",
                (object_type, object_id, record["version"]),
            ).fetchone()
        return bool(row and row["entry_signature"])

    def audit_entries(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_chain ORDER BY sequence DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def verify_audit_chain(self) -> bool:
        previous_hash = "0" * 64
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_chain ORDER BY sequence"
            ).fetchall()
            object_count = int(
                connection.execute("SELECT COUNT(*) FROM research_objects").fetchone()[0]
            )
            if object_count != len(rows):
                return False
            objects = {
                (row["object_type"], row["object_id"], row["version"]): row
                for row in connection.execute("SELECT * FROM research_objects").fetchall()
            }
        for row in rows:
            research_object = objects.get(
                (row["object_type"], row["object_id"], row["object_version"])
            )
            if research_object is None:
                return False
            try:
                normalized_payload = _canonical_json(
                    json.loads(research_object["payload_json"])
                )
            except (json.JSONDecodeError, TypeError):
                return False
            object_hash = _sha256(normalized_payload)
            if (
                object_hash != research_object["content_sha256"]
                or object_hash != row["payload_sha256"]
            ):
                return False
            material = _canonical_json(
                {
                    "action": row["action"],
                    "actor": row["actor"],
                    "created_at": row["created_at"],
                    "object_id": row["object_id"],
                    "object_type": row["object_type"],
                    "object_version": row["object_version"],
                    "payload_sha256": row["payload_sha256"],
                    "previous_hash": previous_hash,
                }
            )
            if row["previous_hash"] != previous_hash or row["entry_hash"] != _sha256(material):
                return False
            signature = str(row["entry_signature"] or "")
            if not signature or self._audit_signing_key is None:
                return False
            expected_signature = hmac.new(
                self._audit_signing_key,
                row["entry_hash"].encode("ascii"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(signature, expected_signature):
                return False
            previous_hash = row["entry_hash"]
        return True


__all__ = ["ResearchStore"]
