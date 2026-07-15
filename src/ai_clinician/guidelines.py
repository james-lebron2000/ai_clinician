"""Versioned computable-guideline governance and CPG-on-FHIR export.

``baseline_approved`` and ``ai_candidate`` are deliberately separate release
channels.  An AI candidate can be reviewed and released as a research artifact,
but this module contains no operation that promotes it into the approved
baseline.  All patient-context evaluation is deterministic and fails closed.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from ai_clinician.models import (
    ApplicabilityReport,
    CandidateDiseaseState,
    CausalEffectEstimate,
    ExpertReview,
    GuidelineChannel,
    GuidelineRecommendation,
    GuidelineRelease,
    ModelRelease,
    ModelStatus,
    ReleaseStatus,
    ReviewDecision,
    ReviewRole,
    StateValidationReport,
    TargetTrialResult,
)


class GuidelineGovernanceError(ValueError):
    """Raised when a guideline operation violates a release or safety gate."""


@dataclass(frozen=True, slots=True)
class GuidelineValidationFinding:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class GuidelineValidationReport:
    passed: bool
    findings: tuple[GuidelineValidationFinding, ...]


@dataclass(frozen=True, slots=True)
class FHIRCPGArtifacts:
    plan_definition: Mapping[str, Any]
    activity_definitions: tuple[Mapping[str, Any], ...]
    library: Mapping[str, Any]

    def as_bundle(self) -> dict[str, Any]:
        resources = [self.library, *self.activity_definitions, self.plan_definition]
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": resource} for resource in resources],
        }


@dataclass(frozen=True, slots=True)
class GuidelineEvaluation:
    release_id: str
    reports: tuple[ApplicabilityReport, ...]
    applicable_recommendation_ids: tuple[str, ...]
    abstained: bool
    abstention_reasons: tuple[str, ...]
    conflict_pairs: tuple[tuple[str, str], ...] = ()
    automatic_order_allowed: Literal[False] = False


def _canonical_hash(value: Any, *, exclude: set[str] | None = None) -> str:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json", exclude=exclude or set())
    else:
        payload = value
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def release_review_hash(release: GuidelineRelease) -> str:
    """Hash immutable release content while excluding review workflow fields."""

    return _canonical_hash(
        release,
        exclude={"status", "expert_review_ids", "locked_at", "created_at"},
    )


def validate_recommendation(
    recommendation: GuidelineRecommendation,
) -> GuidelineValidationReport:
    """Run cross-field controls not already enforced by the strict Schema."""

    findings: list[GuidelineValidationFinding] = []
    if recommendation.automatic_order_allowed:
        findings.append(
            GuidelineValidationFinding(
                "AUTOMATIC_ORDER_PROHIBITED", "recommendations cannot create orders"
            )
        )
    if any(option.automatic_order_allowed for option in recommendation.treatment_options):
        findings.append(
            GuidelineValidationFinding(
                "OPTION_ORDER_PROHIBITED", "treatment options must remain advisory"
            )
        )
    if not recommendation.pico.population or not recommendation.pico.outcomes:
        findings.append(
            GuidelineValidationFinding("PICO_INCOMPLETE", "complete PICO is required")
        )
    if not recommendation.eligibility or not recommendation.abstention_conditions:
        findings.append(
            GuidelineValidationFinding(
                "APPLICABILITY_INCOMPLETE",
                "eligibility and explicit abstention conditions are required",
            )
        )
    if not recommendation.safety_constraints:
        findings.append(
            GuidelineValidationFinding(
                "SAFETY_CONSTRAINTS_MISSING", "at least one safety constraint is required"
            )
        )
    if not recommendation.hard_safety_criteria:
        findings.append(
            GuidelineValidationFinding(
                "COMPUTABLE_SAFETY_MISSING",
                "at least one computable hard safety criterion is required",
            )
        )
    for criterion in recommendation.hard_safety_criteria:
        if not _criterion_is_computable(criterion):
            findings.append(
                GuidelineValidationFinding(
                    "UNCOMPUTABLE_SAFETY_CRITERION",
                    f"hard safety criterion is not computable: {criterion}",
                )
            )
    for source in recommendation.sources:
        if not all(
            (
                source.organization,
                source.title,
                source.version,
                source.url,
                source.source_locator,
                source.license_status,
            )
        ):
            findings.append(
                GuidelineValidationFinding(
                    "PROVENANCE_INCOMPLETE",
                    "every source requires organization, version, locator, URL, and license",
                )
            )
    if recommendation.channel is GuidelineChannel.AI_CANDIDATE:
        if not recommendation.research_only:
            findings.append(
                GuidelineValidationFinding(
                    "AI_CANDIDATE_NOT_RESEARCH_ONLY",
                    "AI candidates must remain research-only",
                )
            )
        if not recommendation.candidate_state_ids:
            findings.append(
                GuidelineValidationFinding(
                    "CANDIDATE_STATE_PROVENANCE_MISSING",
                    "AI candidates require at least one candidate state id",
                )
            )
        if any(not option.effect_estimate_ids for option in recommendation.treatment_options):
            findings.append(
                GuidelineValidationFinding(
                    "CAUSAL_EVIDENCE_BINDING_MISSING",
                    "AI candidate treatment options require audited effect estimate ids",
                )
            )
        if len(recommendation.treatment_options) < 2:
            findings.append(
                GuidelineValidationFinding(
                    "INSUFFICIENT_OPTION_SET",
                    "AI candidates must present at least two unranked options",
                )
            )
        if any(
            not option.target_trial_result_ids
            for option in recommendation.treatment_options
        ):
            findings.append(
                GuidelineValidationFinding(
                    "TARGET_TRIAL_BINDING_MISSING",
                    "AI candidate options require locked target-trial result ids",
                )
            )
    return GuidelineValidationReport(passed=not findings, findings=tuple(findings))


_COMPARISON = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.-]*|[\u4e00-\u9fff][\u4e00-\u9fff0-9_.-]*)"
    r"\s*(==|=|!=|<=|>=|<|>)\s*(.+?)\s*$"
)
_MEMBERSHIP = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.-]*|[\u4e00-\u9fff][\u4e00-\u9fff0-9_.-]*)"
    r"\s+(not\s+in|in)\s+(.+?)\s*$",
    re.IGNORECASE,
)


def _criterion_is_computable(criterion: str) -> bool:
    text = criterion.strip()
    return bool(
        _COMPARISON.match(text)
        or _MEMBERSHIP.match(text)
        or (text.startswith("exists(") and text.endswith(")"))
        or text.casefold().startswith("has:")
    )


@dataclass(frozen=True, slots=True)
class _CriterionResult:
    status: Literal["matched", "failed", "missing", "uncomputable"]
    criterion: str


def _context_value(context: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = context
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return False, None
        value = value[component]
    return True, value


def _literal(text: str) -> Any:
    cleaned = text.strip()
    translations = {"true": True, "false": False, "null": None, "none": None}
    if cleaned.casefold() in translations:
        return translations[cleaned.casefold()]
    try:
        return ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        try:
            return float(cleaned) if "." in cleaned else int(cleaned)
        except ValueError:
            return cleaned.strip("'\"")


def _evaluate_criterion(
    criterion: str, context: Mapping[str, Any]
) -> _CriterionResult:
    text = criterion.strip()
    if text.startswith("exists(") and text.endswith(")"):
        field = text[7:-1].strip()
        exists, value = _context_value(context, field)
        return _CriterionResult(
            "matched" if exists and value is not None else "missing", criterion
        )
    if text.casefold().startswith("has:"):
        field = text[4:].strip()
        exists, value = _context_value(context, field)
        return _CriterionResult(
            "matched" if exists and bool(value) else "failed", criterion
        )

    membership = _MEMBERSHIP.match(text)
    if membership:
        field, operation, raw_expected = membership.groups()
        exists, observed = _context_value(context, field)
        if not exists or observed is None:
            return _CriterionResult("missing", criterion)
        expected = _literal(raw_expected)
        if not isinstance(expected, (list, tuple, set, frozenset)):
            return _CriterionResult("uncomputable", criterion)
        matches = observed in expected
        if operation.casefold().replace("  ", " ") == "not in":
            matches = not matches
        return _CriterionResult("matched" if matches else "failed", criterion)

    comparison = _COMPARISON.match(text)
    if not comparison:
        return _CriterionResult("uncomputable", criterion)
    field, operation, raw_expected = comparison.groups()
    exists, observed = _context_value(context, field)
    if not exists or observed is None:
        return _CriterionResult("missing", criterion)
    expected = _literal(raw_expected)
    try:
        if operation in {"=", "=="}:
            matches = observed == expected
        elif operation == "!=":
            matches = observed != expected
        elif operation == "<":
            matches = observed < expected
        elif operation == "<=":
            matches = observed <= expected
        elif operation == ">":
            matches = observed > expected
        else:
            matches = observed >= expected
    except TypeError:
        return _CriterionResult("uncomputable", criterion)
    return _CriterionResult("matched" if matches else "failed", criterion)


def evaluate_recommendation_applicability(
    recommendation: GuidelineRecommendation,
    context: Mapping[str, Any],
) -> ApplicabilityReport:
    """Evaluate structured criteria and abstain on every ambiguity or hazard."""

    matched: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    abstention: list[str] = []

    for criterion in recommendation.eligibility:
        result = _evaluate_criterion(criterion, context)
        if result.status == "matched":
            matched.append(criterion)
        elif result.status == "failed":
            failed.append(criterion)
            abstention.append("ELIGIBILITY_NOT_MET")
        elif result.status == "missing":
            missing.append(criterion)
            abstention.append("MISSING_CRITICAL_FEATURE")
        else:
            failed.append(criterion)
            abstention.append("UNCOMPUTABLE_ELIGIBILITY_CRITERION")

    for exclusion in recommendation.exclusions:
        result = _evaluate_criterion(exclusion, context)
        if result.status == "matched":
            failed.append(f"exclusion: {exclusion}")
            abstention.append("EXCLUSION_CRITERION_MET")
        elif result.status == "missing":
            missing.append(f"exclusion: {exclusion}")
            abstention.append("MISSING_EXCLUSION_FEATURE")
        elif result.status == "uncomputable":
            failed.append(f"exclusion: {exclusion}")
            abstention.append("UNCOMPUTABLE_EXCLUSION_CRITERION")
        else:
            matched.append(f"exclusion absent: {exclusion}")

    for criterion in recommendation.hard_safety_criteria:
        result = _evaluate_criterion(criterion, context)
        if result.status == "matched":
            matched.append(f"safety satisfied: {criterion}")
        elif result.status == "failed":
            failed.append(f"safety failed: {criterion}")
            abstention.append("HARD_SAFETY_CRITERION_NOT_MET")
        elif result.status == "missing":
            missing.append(f"safety: {criterion}")
            abstention.append("MISSING_HARD_SAFETY_FEATURE")
        else:
            failed.append(f"safety uncomputable: {criterion}")
            abstention.append("UNCOMPUTABLE_HARD_SAFETY_CRITERION")

    for condition in recommendation.abstention_conditions:
        result = _evaluate_criterion(condition, context)
        if result.status == "matched":
            failed.append(f"abstention: {condition}")
            abstention.append("ABSTENTION_CONDITION_MET")
        elif result.status == "missing":
            missing.append(f"abstention: {condition}")
            abstention.append("MISSING_ABSTENTION_FEATURE")
        elif result.status == "uncomputable":
            failed.append(f"abstention: {condition}")
            abstention.append("UNCOMPUTABLE_ABSTENTION_CONDITION")
        else:
            matched.append(f"abstention absent: {condition}")

    ood_status_known = "ood" in context and context.get("ood") is not None
    ood = bool(context.get("ood")) if ood_status_known else True
    if not ood_status_known:
        missing.append("ood")
        abstention.append("OOD_STATUS_MISSING")
    elif ood:
        abstention.append("OUT_OF_DISTRIBUTION")
    conflict_status_known = (
        ("timeline_conflicts" in context and context.get("timeline_conflicts") is not None)
        or ("timeline_conflict" in context and context.get("timeline_conflict") is not None)
    )
    if "timeline_conflicts" in context:
        timeline_conflicts = context.get("timeline_conflicts") or ()
    elif context.get("timeline_conflict"):
        timeline_conflicts = ("timeline_conflict",)
    else:
        timeline_conflicts = ()
    if not conflict_status_known:
        missing.append("timeline_conflicts")
        abstention.append("TIMELINE_CONFLICT_STATUS_MISSING")
    if timeline_conflicts:
        failed.append("timeline_conflicts")
        abstention.append("TIMELINE_CONFLICT")
    unique_reasons = list(dict.fromkeys(abstention))
    applicable = not unique_reasons
    source_basis = [
        f"{source.organization} {source.version} {source.source_locator} "
        f"({source.license_status})"
        for source in recommendation.sources
    ]
    return ApplicabilityReport(
        recommendation_id=recommendation.id,
        applicable=applicable,
        matched_criteria=matched,
        failed_criteria=failed,
        missing_critical_features=missing,
        ood=ood,
        abstained=not applicable,
        abstention_reasons=unique_reasons,
        independent_review_basis=source_basis,
        automatic_order_allowed=False,
    )


def _fhir_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9.-]", "-", value).strip("-.")
    return (cleaned or "artifact")[:64]


def _cql_for(recommendation: GuidelineRecommendation) -> str:
    cql_version = recommendation.guideline_version.replace("'", "''")
    lines = [
        f"library {_fhir_id(recommendation.id).replace('-', '_')} version "
        f"'{cql_version}'",
        "using FHIR version '4.0.1'",
        "context Patient",
        "",
        "/* Eligibility is evaluated by the deterministic host engine.",
        "   This library is research-only and cannot instantiate an order. */",
    ]
    for index, criterion in enumerate(recommendation.eligibility, start=1):
        escaped = criterion.replace("*/", "* /")
        lines.append(f"/* eligibility_{index}: {escaped} */")
    for index, criterion in enumerate(recommendation.exclusions, start=1):
        escaped = criterion.replace("*/", "* /")
        lines.append(f"/* exclusion_{index}: {escaped} */")
    for index, criterion in enumerate(recommendation.hard_safety_criteria, start=1):
        escaped = criterion.replace("*/", "* /")
        lines.append(f"/* hard_safety_{index}: {escaped} */")
    lines.extend(
        [
            "define \"AutomaticOrderAllowed\": false",
            "define \"RequiresIndependentClinicianReview\": true",
            "/* Host criteria are not yet mapped to FHIR search expressions. */",
            "define \"RecommendationApplicable\": false",
        ]
    )
    return "\n".join(lines) + "\n"


def export_cpg_on_fhir(
    recommendation: GuidelineRecommendation,
    *,
    canonical_base: str = "https://example.invalid/fhir/ai-clinician",
) -> FHIRCPGArtifacts:
    """Export FHIR R4 CPG artifacts without an executable order resource."""

    validation = validate_recommendation(recommendation)
    if not validation.passed:
        codes = ", ".join(finding.code for finding in validation.findings)
        raise GuidelineGovernanceError(f"recommendation is not exportable: {codes}")
    rec_id = _fhir_id(recommendation.id)
    status = (
        "active"
        if recommendation.status in {ReleaseStatus.LOCKED, ReleaseStatus.RELEASED}
        else "draft"
    )
    library_id = _fhir_id(f"library-{rec_id}")
    plan_id = _fhir_id(f"plan-{rec_id}")
    cql = _cql_for(recommendation)
    provenance_extensions = [
        {
            "url": "source-provenance",
            "extension": [
                {"url": "organization", "valueString": source.organization},
                {"url": "version", "valueString": source.version},
                {"url": "locator", "valueString": source.source_locator},
                {"url": "license", "valueString": source.license_status},
                {"url": "url", "valueUri": source.url},
            ],
        }
        for source in recommendation.sources
    ]
    library = {
        "resourceType": "Library",
        "id": library_id,
        "url": f"{canonical_base}/Library/{library_id}",
        "version": recommendation.guideline_version,
        "name": _fhir_id(f"logic-{rec_id}"),
        "status": status,
        "experimental": True,
        "type": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/library-type",
                    "code": "logic-library",
                }
            ]
        },
        "extension": [
            {
                "url": "research-only",
                "valueBoolean": True,
            },
            {
                "url": "automatic-order-allowed",
                "valueBoolean": False,
            },
            *provenance_extensions,
        ],
        "content": [
            {
                "contentType": "text/cql",
                "data": base64.b64encode(cql.encode("utf-8")).decode("ascii"),
            }
        ],
    }

    activities: list[Mapping[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for option in recommendation.treatment_options:
        activity_id = _fhir_id(f"activity-{rec_id}-{option.id}")
        activity = {
            "resourceType": "ActivityDefinition",
            "id": activity_id,
            "url": f"{canonical_base}/ActivityDefinition/{activity_id}",
            "version": recommendation.guideline_version,
            "name": _fhir_id(f"option-{option.id}"),
            "title": option.regimen_class,
            "status": status,
            "experimental": True,
            "description": (
                "Research option for independent clinician review; not an order."
            ),
            "extension": [
                {"url": "research-only", "valueBoolean": True},
                {"url": "automatic-order-allowed", "valueBoolean": False},
                {"url": "pareto-optimal", "valueBoolean": option.pareto_optimal},
            ],
        }
        activities.append(activity)
        actions.append(
            {
                "id": _fhir_id(f"action-{option.id}"),
                "title": f"Consider {option.regimen_class}",
                "description": "Present as an option for independent review only.",
                "definitionCanonical": activity["url"],
                "condition": [
                    {
                        "kind": "applicability",
                        "expression": {
                            "language": "text/cql-identifier",
                            "expression": "RecommendationApplicable",
                        },
                    }
                ],
                "extension": [
                    {"url": "automatic-order-allowed", "valueBoolean": False}
                ],
            }
        )
    plan = {
        "resourceType": "PlanDefinition",
        "id": plan_id,
        "url": f"{canonical_base}/PlanDefinition/{plan_id}",
        "version": recommendation.guideline_version,
        "name": _fhir_id(f"guideline-{rec_id}"),
        "title": recommendation.title,
        "status": status,
        "experimental": True,
        "type": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/plan-definition-type",
                    "code": "clinical-protocol",
                }
            ]
        },
        "library": [library["url"]],
        "extension": [
            {"url": "guideline-channel", "valueCode": recommendation.channel.value},
            {"url": "research-only", "valueBoolean": True},
            {"url": "automatic-order-allowed", "valueBoolean": False},
            {
                "url": "grade-evidence-to-decision-sha256",
                "valueString": _canonical_hash(recommendation.evidence_to_decision),
            },
        ],
        "action": actions,
    }
    return FHIRCPGArtifacts(
        plan_definition=plan,
        activity_definitions=tuple(activities),
        library=library,
    )


class GuidelineRegistry:
    """In-memory deterministic registry; persistence belongs to the audit store."""

    def __init__(self) -> None:
        self._recommendations: dict[str, GuidelineRecommendation] = {}
        self._releases: dict[str, GuidelineRelease] = {}
        self._reviews: dict[str, ExpertReview] = {}
        self._candidate_states: dict[str, CandidateDiseaseState] = {}
        self._candidate_state_reports: dict[str, StateValidationReport] = {}
        self._causal_effects: dict[str, CausalEffectEstimate] = {}
        self._causal_results: dict[str, TargetTrialResult] = {}

    def register_candidate_state(
        self,
        state: CandidateDiseaseState,
        *,
        model_release: ModelRelease,
        validation_report: StateValidationReport,
        state_reviews: Sequence[ExpertReview],
        model_reviews: Sequence[ExpertReview],
    ) -> CandidateDiseaseState:
        if state.id in self._candidate_states:
            raise GuidelineGovernanceError("candidate state id is already registered")
        if not state.eligible_for_guideline:
            raise GuidelineGovernanceError(
                "candidate state has not passed stability, calibration, gain, and expert gates"
            )
        if model_release.status not in {ModelStatus.LOCKED, ModelStatus.RELEASED}:
            raise GuidelineGovernanceError("candidate state model is not locked")
        if state.model_release_id != model_release.id:
            raise GuidelineGovernanceError("candidate state model release mismatch")
        if state.model_release_hash != model_release.content_hash():
            raise GuidelineGovernanceError("candidate state model release hash mismatch")
        if (
            state.validation_report_id != validation_report.id
            or state.validation_report_hash != validation_report.content_hash()
            or state.id not in validation_report.candidate_state_ids
            or validation_report.model_release_id != model_release.id
            or validation_report.model_release_hash != model_release.content_hash()
            or validation_report.id not in model_release.validation_report_ids
            or validation_report.data_snapshot_id not in model_release.data_snapshot_ids
            or not validation_report.passed
        ):
            raise GuidelineGovernanceError(
                "candidate state does not bind the locked validation report"
            )
        self._validate_bound_reviews(
            object_id=model_release.id,
            object_hash=model_release.content_hash(),
            review_ids=model_release.expert_review_ids,
            reviews=model_reviews,
            required_roles={
                ReviewRole.CRC_CLINICAL_EXPERT,
                ReviewRole.METHODOLOGIST,
            },
        )
        self._validate_bound_reviews(
            object_id=state.id,
            object_hash=state.content_hash(),
            review_ids=state.expert_review_ids,
            reviews=state_reviews,
            required_roles={ReviewRole.CRC_CLINICAL_EXPERT},
            minimum_reviewers=2,
            require_all_roles={ReviewRole.CRC_CLINICAL_EXPERT},
        )
        if (
            state.expert_confirmations != len({review.reviewer_key for review in state_reviews})
            or validation_report.expert_confirmations != state.expert_confirmations
            or set(validation_report.expert_review_ids)
            != set(state.expert_review_ids)
        ):
            raise GuidelineGovernanceError(
                "candidate state expert confirmation count is not review-bound"
            )
        self._candidate_states[state.id] = state.model_copy(deep=True)
        self._candidate_state_reports[state.id] = validation_report.model_copy(deep=True)
        return state.model_copy(deep=True)

    @staticmethod
    def _validate_bound_reviews(
        *,
        object_id: str,
        object_hash: str,
        review_ids: Sequence[str],
        reviews: Sequence[ExpertReview],
        required_roles: set[ReviewRole],
        minimum_reviewers: int = 2,
        require_all_roles: set[ReviewRole] | None = None,
    ) -> None:
        if set(review_ids) != {review.id for review in reviews}:
            raise GuidelineGovernanceError("review ids do not bind supplied reviews")
        if len({review.reviewer_key for review in reviews}) < minimum_reviewers:
            raise GuidelineGovernanceError("reviews are not independently authored")
        if any(
            review.object_id != object_id
            or review.object_hash != object_hash
            or review.decision is not ReviewDecision.APPROVE
            or not review.independent
            for review in reviews
        ):
            raise GuidelineGovernanceError("artifact has an invalid bound review")
        roles = {review.role for review in reviews}
        if require_all_roles is not None:
            if any(review.role not in require_all_roles for review in reviews):
                raise GuidelineGovernanceError("artifact review has an invalid role")
        elif not required_roles.issubset(roles):
            raise GuidelineGovernanceError("artifact review roles are incomplete")

    def has_candidate_state(self, state_id: str) -> bool:
        return state_id in self._candidate_states

    def register_causal_result(
        self, result: TargetTrialResult
    ) -> TargetTrialResult:
        if result.id in self._causal_results:
            raise GuidelineGovernanceError("target-trial result id is already registered")
        for effect in result.effects:
            if effect.id in self._causal_effects:
                raise GuidelineGovernanceError("causal effect id is already registered")
            sensitivity_passed = (
                str(effect.sensitivity_analysis.get("passed", "")).casefold() == "true"
            )
            gates = (
                not effect.abstained,
                bool(effect.corroborating_methods),
                effect.ci_lower is not None,
                effect.ci_upper is not None,
                effect.ci_lower is not None
                and effect.ci_upper is not None
                and effect.ci_upper - effect.ci_lower <= 1.0,
                effect.eligible_patients >= 1_000,
                effect.strategy_patients >= 200,
                effect.comparator_patients >= 200,
                effect.common_support_fraction >= 0.80,
                effect.strategy_effective_sample_size >= 100,
                effect.comparator_effective_sample_size >= 100,
                effect.maximum_absolute_smd < 0.10,
                effect.negative_controls_passed,
                effect.directionally_consistent,
                sensitivity_passed,
                effect.target_trial_result_id == result.id,
                effect.target_trial_spec_hash == result.spec.content_hash(),
            )
            if not all(gates):
                raise GuidelineGovernanceError(
                    "causal estimate has not passed the locked target-trial gates"
                )
        for effect in result.effects:
            self._causal_effects[effect.id] = effect.model_copy(deep=True)
        self._causal_results[result.id] = result.model_copy(deep=True)
        return result.model_copy(deep=True)

    def has_causal_effect(self, effect_id: str) -> bool:
        return effect_id in self._causal_effects

    def has_causal_result(self, result_id: str) -> bool:
        return result_id in self._causal_results

    def register_recommendation(
        self, recommendation: GuidelineRecommendation
    ) -> GuidelineRecommendation:
        validation = validate_recommendation(recommendation)
        if not validation.passed:
            details = ", ".join(f.code for f in validation.findings)
            raise GuidelineGovernanceError(f"recommendation failed validation: {details}")
        if recommendation.channel is GuidelineChannel.AI_CANDIDATE:
            if any(
                state_id not in self._candidate_states
                for state_id in recommendation.candidate_state_ids
            ):
                raise GuidelineGovernanceError(
                    "AI candidate references an unvalidated disease state"
                )
            if any(
                self._candidate_states[state_id].content_hash()
                != recommendation.candidate_state_hashes.get(state_id)
                for state_id in recommendation.candidate_state_ids
            ):
                raise GuidelineGovernanceError(
                    "AI candidate state content hash does not match"
                )
            effect_ids = {
                effect_id
                for option in recommendation.treatment_options
                for effect_id in option.effect_estimate_ids
            }
            if any(effect_id not in self._causal_effects for effect_id in effect_ids):
                raise GuidelineGovernanceError(
                    "AI candidate references an unaudited causal estimate"
                )
            for option in recommendation.treatment_options:
                for effect_id in option.effect_estimate_ids:
                    effect = self._causal_effects[effect_id]
                    if effect.content_hash() != option.effect_estimate_hashes.get(effect_id):
                        raise GuidelineGovernanceError(
                            "AI candidate effect content hash does not match"
                        )
                    if option.regimen_class not in {effect.strategy, effect.comparator}:
                        raise GuidelineGovernanceError(
                            "treatment option does not match the target-trial contrast"
                        )
                    if effect.target_trial_result_id not in option.target_trial_result_ids:
                        raise GuidelineGovernanceError(
                            "effect is not bound to the option target-trial result"
                        )
                    result = self._causal_results.get(effect.target_trial_result_id or "")
                    if result is None:
                        raise GuidelineGovernanceError(
                            "AI candidate references an unregistered target-trial result"
                        )
                    if result.spec.decision_point is not option.decision_point:
                        raise GuidelineGovernanceError(
                            "target-trial decision point does not match treatment option"
                        )
                for result_id in option.target_trial_result_ids:
                    result = self._causal_results.get(result_id)
                    if result is None or result.content_hash() != option.target_trial_result_hashes.get(
                        result_id
                    ):
                        raise GuidelineGovernanceError(
                            "AI candidate target-trial result hash does not match"
                        )
            result_ids = {
                result_id
                for option in recommendation.treatment_options
                for result_id in option.target_trial_result_ids
            }
            results = [self._causal_results[result_id] for result_id in result_ids]
            study_ids = {result.spec.study_id for result in results}
            cohort_ids = {result.spec.cohort_snapshot_id for result in results}
            data_snapshot_ids = {result.spec.data_snapshot_id for result in results}
            state_snapshot_ids = {
                self._candidate_state_reports[state_id].data_snapshot_id
                for state_id in recommendation.candidate_state_ids
            }
            if (
                len(study_ids) != 1
                or len(cohort_ids) != 1
                or len(data_snapshot_ids) != 1
                or data_snapshot_ids != state_snapshot_ids
                or cohort_ids != data_snapshot_ids
            ):
                raise GuidelineGovernanceError(
                    "AI candidate evidence is not bound to one locked study cohort"
                )
        if recommendation.id in self._recommendations:
            raise GuidelineGovernanceError("recommendation id is already registered")
        # Store a deep copy so caller-side mutation cannot silently alter a
        # content-addressed release.
        stored = recommendation.model_copy(deep=True)
        self._recommendations[stored.id] = stored
        return stored.model_copy(deep=True)

    def get_recommendation(self, recommendation_id: str) -> GuidelineRecommendation:
        try:
            return self._recommendations[recommendation_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"unknown recommendation {recommendation_id!r}") from exc

    def create_release(
        self,
        *,
        name: str,
        version: str,
        channel: GuidelineChannel,
        recommendation_ids: Sequence[str],
        context_of_use: str,
    ) -> GuidelineRelease:
        if not recommendation_ids or len(set(recommendation_ids)) != len(
            recommendation_ids
        ):
            raise GuidelineGovernanceError(
                "a release requires unique recommendation ids"
            )
        recommendations = [
            self._recommendations.get(item) for item in recommendation_ids
        ]
        if any(item is None for item in recommendations):
            raise GuidelineGovernanceError("release references an unknown recommendation")
        typed = [item for item in recommendations if item is not None]
        if any(item.channel is not channel for item in typed):
            raise GuidelineGovernanceError(
                "baseline and AI-candidate recommendations cannot share a release"
            )
        release = GuidelineRelease(
            name=name,
            version=version,
            channel=channel,
            status=ReleaseStatus.IN_REVIEW,
            recommendation_ids=list(recommendation_ids),
            recommendation_hashes={item.id: item.content_hash() for item in typed},
            context_of_use=context_of_use,
            research_only=True,
            automatic_order_allowed=False,
        )
        self._releases[release.id] = release
        return release.model_copy(deep=True)

    def get_release(self, release_id: str) -> GuidelineRelease:
        try:
            return self._releases[release_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"unknown guideline release {release_id!r}") from exc

    def restore_release(self, release: GuidelineRelease) -> GuidelineRelease:
        """Hydrate an audited release while rechecking every bound recommendation.

        This is intentionally narrower than a general import API and exists so
        the local CLI can resume a review workflow from the append-only store.
        """

        if release.id in self._releases:
            raise GuidelineGovernanceError("release id is already registered")
        if set(release.recommendation_ids) != set(release.recommendation_hashes):
            raise GuidelineGovernanceError("release recommendation hashes are incomplete")
        for recommendation_id, expected_hash in release.recommendation_hashes.items():
            recommendation = self._recommendations.get(recommendation_id)
            if recommendation is None:
                raise GuidelineGovernanceError(
                    "restored release references an unknown recommendation"
                )
            if recommendation.channel is not release.channel:
                raise GuidelineGovernanceError("restored release crosses guideline channels")
            if recommendation.content_hash() != expected_hash:
                raise GuidelineGovernanceError(
                    "restored release does not match recommendation content"
                )
        stored = GuidelineRelease.model_validate(release.model_dump())
        self._releases[stored.id] = stored.model_copy(deep=True)
        return stored.model_copy(deep=True)

    def expected_review_hash(self, object_id: str) -> str:
        if object_id in self._recommendations:
            return self._recommendations[object_id].content_hash()
        if object_id in self._releases:
            return release_review_hash(self._releases[object_id])
        raise KeyError(f"unknown review object {object_id!r}")

    def submit_review(self, review: ExpertReview) -> ExpertReview:
        if review.id in self._reviews:
            raise GuidelineGovernanceError("review id is already registered")
        expected = self.expected_review_hash(review.object_id)
        if review.object_hash != expected:
            raise GuidelineGovernanceError("review does not bind the current artifact hash")
        if not review.independent:
            raise GuidelineGovernanceError("guideline reviews must be independent")
        duplicate = any(
            existing.object_id == review.object_id
            and existing.reviewer_key == review.reviewer_key
            for existing in self._reviews.values()
        )
        if duplicate:
            raise GuidelineGovernanceError(
                "one reviewer can submit only one review per artifact version"
            )
        stored = review.model_copy(deep=True)
        self._reviews[stored.id] = stored
        return stored.model_copy(deep=True)

    def _approved_reviews(self, release: GuidelineRelease) -> list[ExpertReview]:
        reviews = [
            review for review in self._reviews.values() if review.object_id == release.id
        ]
        if any(review.decision is not ReviewDecision.APPROVE for review in reviews):
            raise GuidelineGovernanceError(
                "a rejection or change request prevents guideline lock"
            )
        return reviews

    def lock_release(self, release_id: str) -> GuidelineRelease:
        release = self._releases.get(release_id)
        if release is None:
            raise KeyError(release_id)
        if release.status is not ReleaseStatus.IN_REVIEW:
            raise GuidelineGovernanceError("only an in-review release can be locked")
        for recommendation_id, bound_hash in release.recommendation_hashes.items():
            recommendation = self._recommendations[recommendation_id]
            if recommendation.content_hash() != bound_hash:
                raise GuidelineGovernanceError(
                    "recommendation content changed after release creation"
                )
        reviews = self._approved_reviews(release)
        roles = {review.role for review in reviews}
        reviewers = {review.reviewer_key for review in reviews}
        if release.channel is GuidelineChannel.BASELINE_APPROVED:
            clinical_count = sum(
                review.role is ReviewRole.CRC_CLINICAL_EXPERT for review in reviews
            )
            if (
                len(reviewers) < 3
                or clinical_count < 2
                or not roles & {ReviewRole.METHODOLOGIST, ReviewRole.SAFETY_REVIEWER}
            ):
                raise GuidelineGovernanceError(
                    "baseline release requires an expert committee: two CRC experts "
                    "and an independent method/safety reviewer"
                )
        else:
            if (
                len(reviewers) < 2
                or ReviewRole.CRC_CLINICAL_EXPERT not in roles
                or ReviewRole.METHODOLOGIST not in roles
            ):
                raise GuidelineGovernanceError(
                    "AI candidate lock requires clinical and methodological approval"
                )
        locked = release.model_copy(
            update={
                "status": ReleaseStatus.LOCKED,
                "expert_review_ids": [review.id for review in reviews],
                "locked_at": datetime.now(timezone.utc),
            },
            deep=True,
        )
        # model_copy does not revalidate in Pydantic v2; explicitly validate the
        # final release contract before storing it.
        locked = GuidelineRelease.model_validate(locked.model_dump())
        self._releases[release_id] = locked
        return locked.model_copy(deep=True)

    def publish_release(
        self, release_id: str, *, publisher_role: str
    ) -> GuidelineRelease:
        release = self._releases.get(release_id)
        if release is None:
            raise KeyError(release_id)
        if release.status is not ReleaseStatus.LOCKED:
            raise GuidelineGovernanceError("only a locked release can be published")
        if release.channel is GuidelineChannel.BASELINE_APPROVED:
            if publisher_role != "expert_committee":
                raise GuidelineGovernanceError(
                    "only the expert committee can publish baseline guidance"
                )
        elif publisher_role not in {"research_governance", "expert_committee"}:
            raise GuidelineGovernanceError(
                "AI candidates require research-governance publication"
            )
        published = GuidelineRelease.model_validate(
            release.model_copy(update={"status": ReleaseStatus.RELEASED}).model_dump()
        )
        self._releases[release_id] = published
        return published.model_copy(deep=True)

    def promote_ai_candidate_to_baseline(self, release_id: str) -> None:
        release = self._releases.get(release_id)
        if release is None:
            raise KeyError(release_id)
        raise GuidelineGovernanceError(
            "automatic AI-candidate promotion is prohibited; create a separately "
            "sourced and committee-approved baseline recommendation"
        )

    def evaluate(
        self,
        release_id: str,
        context: Mapping[str, Any],
        *,
        explicit_conflicts: Sequence[tuple[str, str]] = (),
    ) -> GuidelineEvaluation:
        release = self._releases.get(release_id)
        if release is None:
            raise KeyError(release_id)
        if release.status not in {ReleaseStatus.LOCKED, ReleaseStatus.RELEASED}:
            raise GuidelineGovernanceError("only locked guidance can be evaluated")
        reports = [
            evaluate_recommendation_applicability(
                self._recommendations[recommendation_id], context
            )
            for recommendation_id in release.recommendation_ids
        ]
        applicable = {report.recommendation_id for report in reports if report.applicable}
        conflicts = tuple(
            sorted(
                {
                    tuple(sorted((left, right)))
                    for left, right in explicit_conflicts
                    if left in applicable and right in applicable and left != right
                }
            )
        )
        if conflicts:
            conflicted_ids = {item for pair in conflicts for item in pair}
            reports = [
                ApplicabilityReport.model_validate(
                    report.model_copy(
                        update={
                            "applicable": False,
                            "abstained": True,
                            "failed_criteria": [
                                *report.failed_criteria,
                                "conflicting guideline recommendation",
                            ],
                            "abstention_reasons": [
                                *report.abstention_reasons,
                                "GUIDELINE_CONFLICT",
                            ],
                        }
                    ).model_dump()
                )
                if report.recommendation_id in conflicted_ids
                else report
                for report in reports
            ]
            applicable -= conflicted_ids
        reasons: list[str] = []
        if conflicts:
            reasons.append("GUIDELINE_CONFLICT")
        if not applicable:
            reasons.append("NO_UNAMBIGUOUS_APPLICABLE_RECOMMENDATION")
        return GuidelineEvaluation(
            release_id=release_id,
            reports=tuple(reports),
            applicable_recommendation_ids=tuple(sorted(applicable)),
            abstained=bool(reasons),
            abstention_reasons=tuple(reasons),
            conflict_pairs=conflicts,
        )


GuidelineEngine = GuidelineRegistry
to_cpg_on_fhir = export_cpg_on_fhir


__all__ = [
    "FHIRCPGArtifacts",
    "GuidelineEngine",
    "GuidelineEvaluation",
    "GuidelineGovernanceError",
    "GuidelineRegistry",
    "GuidelineValidationFinding",
    "GuidelineValidationReport",
    "evaluate_recommendation_applicability",
    "export_cpg_on_fhir",
    "release_review_hash",
    "to_cpg_on_fhir",
    "validate_recommendation",
]
