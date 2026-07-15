"""Aggregate-only FastAPI surface for the AI Clinician research platform."""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import Field

from .guidelines import release_review_hash
from .feature_manifest import AnalysisManifestRegistration
from .identity import GovernanceAuthenticationError, GovernanceDirectory
from .privacy import is_placeholder_secret
from .models import (
    CandidateDiseaseState,
    EvidenceCertainty,
    ExpertReview,
    GuidelineChannel,
    GuidelineRecommendation,
    GuidelineRelease,
    LifecycleStatus,
    Jurisdiction,
    ModelRelease,
    ReviewDecision,
    ReviewRole,
    RecommendationStrength,
    ReleaseStatus,
    StrictModel,
    TimelineValidationManifestRegistration,
    ResearchStudy,
)
from .store import ResearchStore


def _authorizer(expected_key: str | None) -> Callable[..., None]:
    def authorize(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key is None or not hmac.compare_digest(x_api_key, expected_key or ""):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid research API key required",
            )

    return authorize


class ExpertReviewSubmission(StrictModel):
    object_id: str = Field(min_length=3)
    object_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    role: ReviewRole
    decision: ReviewDecision
    rationale: str = Field(min_length=10)


class StoredObjectMetadata(StrictModel):
    object_id: str
    version: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    created_at: datetime


class StudySummaryView(StoredObjectMetadata):
    state: LifecycleStatus
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
    research_only: bool = True


class GuidelineReleaseView(StoredObjectMetadata):
    release_id: str
    channel: GuidelineChannel
    status: ReleaseStatus
    recommendation_count: int = Field(ge=1)
    expert_review_count: int = Field(ge=0)
    locked: bool
    research_only: bool
    automatic_order_allowed: bool


class GuidelineRecommendationView(StoredObjectMetadata):
    recommendation_id: str
    channel: GuidelineChannel
    jurisdiction: Jurisdiction
    evidence_certainty: EvidenceCertainty
    recommendation_strength: RecommendationStrength
    status: ReleaseStatus
    source_count: int = Field(ge=1)
    safety_constraint_count: int = Field(ge=1)
    treatment_option_count: int = Field(ge=0)
    research_only: bool
    automatic_order_allowed: bool


def _metadata(record: dict) -> dict:
    return {
        "object_id": record["object_id"],
        "version": record["version"],
        "content_sha256": record["content_sha256"],
        "created_at": record["created_at"],
    }


def _study_view(record: dict) -> StudySummaryView:
    study = ResearchStudy.model_validate(record["payload"])
    return StudySummaryView(
        **_metadata(record),
        state=study.state,
        data_profile_id=study.data_profile_id,
        timeline_quality_report_id=study.timeline_quality_report_id,
        cohort_snapshot_id=study.cohort_snapshot_id,
        state_model_release_id=study.state_model_release_id,
        target_trial_ids=study.target_trial_ids,
        guideline_release_id=study.guideline_release_id,
        guideline_release_version=study.guideline_release_version,
        guideline_release_hash=study.guideline_release_hash,
    )


def _release_view(record: dict) -> GuidelineReleaseView:
    release = GuidelineRelease.model_validate(record["payload"])
    return GuidelineReleaseView(
        **_metadata(record),
        release_id=release.id,
        channel=release.channel,
        status=release.status,
        recommendation_count=len(release.recommendation_ids),
        expert_review_count=len(release.expert_review_ids),
        locked=release.status in {ReleaseStatus.LOCKED, ReleaseStatus.RELEASED},
        research_only=release.research_only,
        automatic_order_allowed=release.automatic_order_allowed,
    )


def _recommendation_view(record: dict) -> GuidelineRecommendationView:
    recommendation = GuidelineRecommendation.model_validate(record["payload"])
    return GuidelineRecommendationView(
        **_metadata(record),
        recommendation_id=recommendation.id,
        channel=recommendation.channel,
        jurisdiction=recommendation.jurisdiction,
        evidence_certainty=recommendation.evidence_certainty,
        recommendation_strength=recommendation.recommendation_strength,
        status=recommendation.status,
        source_count=len(recommendation.sources),
        safety_constraint_count=len(recommendation.safety_constraints),
        treatment_option_count=len(recommendation.treatment_options),
        research_only=recommendation.research_only,
        automatic_order_allowed=recommendation.automatic_order_allowed,
    )


def create_app(
    store: ResearchStore,
    *,
    api_key: str | None = None,
    governance_directory: GovernanceDirectory | None = None,
) -> FastAPI:
    """Create an API that has no patient-text, timeline, inference, or order route."""

    expected_key = api_key if api_key is not None else os.environ.get("AI_CLINICIAN_API_KEY")
    if expected_key is None or len(expected_key.encode("utf-8")) < 16:
        raise RuntimeError("AI_CLINICIAN_API_KEY must contain at least 16 bytes")
    if is_placeholder_secret(expected_key):
        raise RuntimeError("AI_CLINICIAN_API_KEY cannot be a placeholder")
    authorize = _authorizer(expected_key)

    def require_valid_audit() -> None:
        if not store.verify_audit_chain():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="research audit chain verification failed",
            )

    governed_dependencies = [Depends(authorize), Depends(require_valid_audit)]
    app = FastAPI(
        title="AI Clinician Research API",
        version="0.1.0",
        description=(
            "Aggregate research results and review metadata only. "
            "No patient-level access, diagnosis, prescription, or order entry."
        ),
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "mode": "research_only",
            "automatic_orders": False,
        }

    @app.get("/v1/studies", dependencies=governed_dependencies, tags=["research"])
    def studies(
        limit: int = Query(default=100, ge=1, le=1_000)
    ) -> list[StudySummaryView]:
        return [_study_view(record) for record in store.list_latest("study", limit=limit)]

    @app.get(
        "/v1/guideline-releases",
        dependencies=governed_dependencies,
        tags=["guidelines"],
    )
    def guideline_releases(
        limit: int = Query(default=100, ge=1, le=1_000),
    ) -> list[GuidelineReleaseView]:
        return [
            _release_view(record)
            for record in store.list_latest("guideline_release", limit=limit)
        ]

    @app.get(
        "/v1/guideline-releases/{release_id}",
        dependencies=governed_dependencies,
        tags=["guidelines"],
    )
    def guideline_release(release_id: str) -> GuidelineReleaseView:
        record = store.get("guideline_release", release_id)
        if record is None:
            raise HTTPException(status_code=404, detail="guideline release not found")
        return _release_view(record)

    @app.get(
        "/v1/recommendations/{recommendation_id}",
        dependencies=governed_dependencies,
        tags=["guidelines"],
    )
    def recommendation(recommendation_id: str) -> GuidelineRecommendationView:
        record = store.get("guideline_recommendation", recommendation_id)
        if record is None:
            raise HTTPException(status_code=404, detail="recommendation not found")
        return _recommendation_view(record)

    @app.post(
        "/v1/expert-reviews",
        status_code=status.HTTP_201_CREATED,
        dependencies=governed_dependencies,
        tags=["review"],
    )
    def create_expert_review(
        submission: ExpertReviewSubmission,
        x_governance_token: str | None = Header(default=None),
    ) -> dict:
        directory = governance_directory
        if directory is None:
            try:
                directory = GovernanceDirectory.from_env()
            except GovernanceAuthenticationError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
        try:
            identity = directory.authenticate(
                x_governance_token, allowed_roles=[submission.role.value]
            )
        except GovernanceAuthenticationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        review = ExpertReview(
            object_id=submission.object_id,
            object_hash=submission.object_hash,
            reviewer_key=identity.subject,
            role=submission.role,
            decision=submission.decision,
            rationale=submission.rationale,
            independent=True,
        )
        recommendation_record = store.get("guideline_recommendation", review.object_id)
        release_record = store.get("guideline_release", review.object_id)
        model_record = store.get("model_release", review.object_id)
        state_record = store.get("candidate_state", review.object_id)
        timeline_manifest_record = store.get(
            "timeline_validation_manifest", review.object_id
        )
        analysis_manifest_record = store.get(
            "analysis_manifest_registration", review.object_id
        )
        if recommendation_record is not None:
            expected_hash = GuidelineRecommendation.model_validate(
                recommendation_record["payload"]
            ).content_hash()
        elif release_record is not None:
            expected_hash = release_review_hash(
                GuidelineRelease.model_validate(release_record["payload"])
            )
        elif model_record is not None:
            expected_hash = ModelRelease.model_validate(
                model_record["payload"]
            ).content_hash()
        elif state_record is not None:
            if submission.role is not ReviewRole.CRC_CLINICAL_EXPERT:
                raise HTTPException(
                    status_code=409,
                    detail="candidate states require CRC clinical expert review",
                )
            expected_hash = CandidateDiseaseState.model_validate(
                state_record["payload"]
            ).content_hash()
        elif timeline_manifest_record is not None:
            if submission.role is not ReviewRole.CRC_CLINICAL_EXPERT:
                raise HTTPException(
                    status_code=409,
                    detail="timeline gold manifest requires CRC clinical expert review",
                )
            expected_hash = TimelineValidationManifestRegistration.model_validate(
                timeline_manifest_record["payload"]
            ).content_hash()
        elif analysis_manifest_record is not None:
            if submission.role is not ReviewRole.CRC_CLINICAL_EXPERT:
                raise HTTPException(
                    status_code=409,
                    detail="state column manifest requires CRC clinical expert review",
                )
            expected_hash = AnalysisManifestRegistration.model_validate(
                analysis_manifest_record["payload"]
            ).content_hash()
        else:
            raise HTTPException(status_code=404, detail="review artifact not found")
        if not review.independent:
            raise HTTPException(status_code=409, detail="review must be independent")
        if not hmac.compare_digest(review.object_hash, expected_hash):
            raise HTTPException(status_code=409, detail="review artifact hash mismatch")
        duplicate = any(
            item["payload"].get("object_id") == review.object_id
            and item["payload"].get("reviewer_key") == review.reviewer_key
            and item["payload"].get("object_hash") == review.object_hash
            for item in store.list_all_latest("expert_review")
        )
        if duplicate:
            raise HTTPException(
                status_code=409,
                detail="reviewer already reviewed this artifact version",
            )
        try:
            stored = store.put(
                "expert_review",
                review.id,
                review,
                actor=identity.subject,
                governance_token=x_governance_token,
            )
        except ValueError as exc:
            if "signed audit" in str(exc):
                raise HTTPException(
                    status_code=503,
                    detail="signed governance audit is not configured",
                ) from exc
            raise
        return {
            **stored,
            "decision": review.decision,
            "automatic_orders": False,
        }

    @app.get("/v1/audit", dependencies=[Depends(authorize)], tags=["audit"])
    def audit(limit: int = Query(default=100, ge=1, le=1_000)) -> list[dict]:
        return store.audit_entries(limit=limit)

    @app.get(
        "/v1/audit/verify", dependencies=[Depends(authorize)], tags=["audit"]
    )
    def verify_audit() -> dict[str, bool]:
        return {"valid": store.verify_audit_chain()}

    return app


__all__ = ["create_app"]
