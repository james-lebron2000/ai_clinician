"""Deterministic, fail-closed lifecycle for the clinical research MVP."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from .models import (
    CohortSnapshot,
    ExpertReview,
    LifecycleStatus,
    MVP_DISABLED_LIFECYCLE_STATES,
    ModelRelease,
    ModelStatus,
    ReviewDecision,
    ReviewRole,
    ResearchStudy,
    StateValidationReport,
    StatePartitionSnapshot,
    TimelineQualityReport,
    TimelineValidationManifestRegistration,
    transition_study,
)
from .store import ResearchStore


class WorkflowGateError(ValueError):
    """Raised when evidence for a lifecycle transition is missing or has failed."""


class ResearchWorkflow:
    """Persists immutable study versions and permits only the fixed state machine."""

    def __init__(
        self, store: ResearchStore, *, governance_token: str | None = None
    ) -> None:
        self.store = store
        self.governance_token = governance_token or os.environ.get(
            "AI_CLINICIAN_GOVERNANCE_TOKEN"
        )

    def create_study(self, name: str, context_of_use: str) -> ResearchStudy:
        study = ResearchStudy(name=name, context_of_use=context_of_use)
        self.store.put(
            "study", study.id, study, governance_token=self.governance_token
        )
        return study

    def get_study(self, study_id: str) -> ResearchStudy:
        record = self.store.get("study", study_id)
        if record is None:
            raise WorkflowGateError("study does not exist")
        return ResearchStudy.model_validate(record["payload"])

    def save_artifact(
        self,
        object_type: str,
        object_id: str,
        artifact: BaseModel | Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.store.put(
            object_type,
            object_id,
            artifact,
            governance_token=self.governance_token,
        )

    def transition(
        self,
        study_id: str,
        target: LifecycleStatus,
        *,
        data_profile_id: str | None = None,
        timeline_quality_report_id: str | None = None,
        cohort_snapshot_id: str | None = None,
        state_model_release_id: str | None = None,
        target_trial_ids: Sequence[str] | None = None,
        guideline_release_id: str | None = None,
    ) -> ResearchStudy:
        study = self.get_study(study_id)
        if target in MVP_DISABLED_LIFECYCLE_STATES:
            # Keep the release-blocking decision ahead of any artifact lookup so
            # callers always receive the explicit fail-closed lifecycle result.
            self._validate_gate(study, target)
        updates: dict[str, Any] = {}
        supplied = {
            "data_profile_id": data_profile_id,
            "timeline_quality_report_id": timeline_quality_report_id,
            "cohort_snapshot_id": cohort_snapshot_id,
            "state_model_release_id": state_model_release_id,
            "guideline_release_id": guideline_release_id,
        }
        updates.update({key: value for key, value in supplied.items() if value is not None})
        if target_trial_ids is not None:
            updates["target_trial_ids"] = list(target_trial_ids)
        if target in {
            LifecycleStatus.CANDIDATE_GUIDELINE,
            LifecycleStatus.RESEARCH_RELEASE,
        }:
            release_id = guideline_release_id or study.guideline_release_id
            release_record = self._require_signed("guideline_release", release_id)
            updates["guideline_release_id"] = release_id
            updates["guideline_release_version"] = release_record["version"]
            updates["guideline_release_hash"] = release_record["content_sha256"]
        candidate = study.model_copy(update=updates)
        self._validate_gate(candidate, target)
        advanced = transition_study(candidate, target)
        self.store.put(
            "study", study_id, advanced, governance_token=self.governance_token
        )
        return advanced

    def _require(self, object_type: str, object_id: str | None) -> dict[str, Any]:
        if not object_id:
            raise WorkflowGateError(f"{object_type} evidence is required")
        record = self.store.get(object_type, object_id)
        if record is None:
            raise WorkflowGateError(f"referenced {object_type} does not exist")
        payload_id = record.get("payload", {}).get("id")
        if payload_id is not None and payload_id != object_id:
            raise WorkflowGateError(
                f"referenced {object_type} payload id does not match its store key"
            )
        return record

    def _require_signed(
        self, object_type: str, object_id: str | None
    ) -> dict[str, Any]:
        record = self._require(object_type, object_id)
        assert object_id is not None
        if not self.store.artifact_has_valid_signature(object_type, object_id):
            raise WorkflowGateError(
                f"referenced {object_type} is not bound to a valid audit signature"
            )
        return record

    def _validate_gate(self, study: ResearchStudy, target: LifecycleStatus) -> None:
        if target is LifecycleStatus.DATA_REGISTERED:
            self._require("data_profile", study.data_profile_id)
        elif target is LifecycleStatus.TIMELINE_VALIDATED:
            record = self._require_signed(
                "timeline_quality_report", study.timeline_quality_report_id
            )
            report = TimelineQualityReport.model_validate(record["payload"])
            if (
                not report.passed
                or report.future_leakage_count
                or not report.provenance_complete()
            ):
                raise WorkflowGateError(
                    "timeline Go/No-Go failed; downgrade to static stratification"
                )
            manifest_record = self._require_signed(
                "timeline_validation_manifest", report.manifest_registration_id
            )
            manifest = TimelineValidationManifestRegistration.model_validate(
                manifest_record["payload"]
            )
            annotation_reviews = [
                ExpertReview.model_validate(
                    self._require_signed("expert_review", review_id)["payload"]
                )
                for review_id in report.annotation_review_ids
            ]
            annotation_reviews_valid = (
                len({review.reviewer_key for review in annotation_reviews}) >= 3
                and all(
                    review.object_id == manifest.id
                    and review.object_hash == manifest.content_hash()
                    and review.role is ReviewRole.CRC_CLINICAL_EXPERT
                    and review.decision is ReviewDecision.APPROVE
                    and review.independent
                    for review in annotation_reviews
                )
            )
            if (
                report.manifest_registration_hash != manifest.content_hash()
                or report.gold_snapshot_id != manifest.gold_snapshot_id
                or report.gold_snapshot_sha256 != manifest.gold_snapshot_sha256
                or report.locked_split_sha256 != manifest.locked_split_sha256
                or report.decision_view_manifest_sha256
                != manifest.decision_view_manifest_sha256
                or report.cohen_kappa != manifest.annotation_kappa
                or not annotation_reviews_valid
            ):
                raise WorkflowGateError(
                    "timeline Go/No-Go failed; downgrade to static stratification"
                )
        elif target is LifecycleStatus.COHORT_LOCKED:
            record = self._require_signed("cohort_snapshot", study.cohort_snapshot_id)
            cohort = CohortSnapshot.model_validate(record["payload"])
            if (
                cohort.study_id != study.id
                or cohort.data_profile_id != study.data_profile_id
            ):
                raise WorkflowGateError("cohort snapshot belongs to another study")
        elif target is LifecycleStatus.STATES_DISCOVERED:
            record = self._require_signed(
                "model_release", study.state_model_release_id
            )
            release = ModelRelease.model_validate(record["payload"])
            if release.status not in {ModelStatus.LOCKED, ModelStatus.RELEASED}:
                raise WorkflowGateError("state model must be locked before discovery")
            for report_id in release.validation_report_ids:
                report_record = self._require_signed(
                    "state_validation_report", report_id
                )
                report = StateValidationReport.model_validate(report_record["payload"])
                cohort = CohortSnapshot.model_validate(
                    self._require_signed("cohort_snapshot", report.data_snapshot_id)[
                        "payload"
                    ]
                )
                partition = StatePartitionSnapshot.model_validate(
                    self._require_signed(
                        "state_partition_snapshot", report.partition_snapshot_id
                    )["payload"]
                )
                if (
                    report.model_release_id != release.id
                    or report.model_release_hash != release.content_hash()
                    or report.data_snapshot_id not in release.data_snapshot_ids
                    or report.data_snapshot_hash != cohort.content_hash()
                    or report.partition_snapshot_hash != partition.content_hash()
                    or partition.cohort_snapshot_id != cohort.id
                    or partition.cohort_snapshot_hash != cohort.content_hash()
                    or report.locked_test_patient_count
                    != partition.locked_test_patient_count
                    or not report.passed
                ):
                    raise WorkflowGateError("state model validation report did not pass")
                checks = (
                    report.bootstrap_ari >= 0.75,
                    0.8 <= report.calibration_slope <= 1.2,
                    report.relative_brier_improvement >= 0.05,
                    report.expert_confirmations >= 2,
                )
                if not all(checks):
                    raise WorkflowGateError("state validation thresholds were not met")
            reviews = [
                ExpertReview.model_validate(
                    self._require_signed("expert_review", review_id)["payload"]
                )
                for review_id in release.expert_review_ids
            ]
            if any(
                review.object_id != release.id
                or review.object_hash != release.content_hash()
                or review.decision is not ReviewDecision.APPROVE
                or not review.independent
                for review in reviews
            ):
                raise WorkflowGateError("state model has an invalid bound review")
            if len({review.reviewer_key for review in reviews}) < 2:
                raise WorkflowGateError("state model reviews are not independent")
            model_roles = {review.role for review in reviews}
            if not {
                ReviewRole.CRC_CLINICAL_EXPERT,
                ReviewRole.METHODOLOGIST,
            }.issubset(model_roles):
                raise WorkflowGateError(
                    "state model requires clinical and methodological review"
                )
        elif target is LifecycleStatus.TARGET_TRIALS_COMPLETED:
            raise WorkflowGateError(
                "target-trial lifecycle is disabled until deterministic causal audits exist"
            )
        elif target is LifecycleStatus.CANDIDATE_GUIDELINE:
            raise WorkflowGateError(
                "AI-candidate promotion is disabled until locked state and causal evaluators exist"
            )
        elif target is LifecycleStatus.RESEARCH_RELEASE:
            raise WorkflowGateError(
                "research release is disabled until locked state and causal evaluators exist"
            )


__all__ = ["ResearchWorkflow", "WorkflowGateError"]
