from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .models import (
    AnalysisReport,
    ControlSpec,
    EvidenceRecord,
    ExperimentPlan,
    Hypothesis,
    Observation,
    PlateAssignment,
    PolicyFinding,
    PolicyReport,
    ProjectSnapshot,
    ResearchGoal,
    ResultBundle,
    StudyPhase,
    StopRules,
    WorkOrder,
)
from .optimizer import ConditionSpace, DesignResult
from .policy import CompliancePolicy


class EvidenceProvider(Protocol):
    def collect(self, goal: ResearchGoal) -> list[dict[str, str | list[str]]]: ...


class ReasoningBackend(Protocol):
    """Replaceable reasoning backend; production adapters must remain inside the approved boundary."""

    def summarize(self, prompt: str, evidence: list[EvidenceRecord]) -> str: ...


class RuleBasedReasoningBackend:
    def summarize(self, prompt: str, evidence: list[EvidenceRecord]) -> str:
        sources = ", ".join(record.title for record in evidence[:3])
        return f"{prompt.strip()} Evidence basis: {sources}."


class CuratedEvidenceProvider:
    """Local, traceable evidence seed. Network retrieval can be added behind the same interface."""

    def collect(self, goal: ResearchGoal) -> list[dict[str, str | list[str]]]:
        return [
            {
                "claim": "NAM validation must be tied to a context of use and include biological relevance, technical characterization and fitness for purpose.",
                "title": "FDA General Considerations for the Use of NAMs in Drug Development",
                "source_url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/general-considerations-use-new-approach-methodologies-drug-development",
                "source_type": "regulatory",
                "supports": ["context_of_use", "quality_system"],
            },
            {
                "claim": "FDA evaluates AI model credibility for a specific context of use using a risk-based framework.",
                "title": "FDA Considerations for AI Supporting Regulatory Decision-Making",
                "source_url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-use-artificial-intelligence-support-regulatory-decision-making-drug-and-biological",
                "source_type": "regulatory",
                "supports": ["ai_context_of_use", "credibility_assessment"],
            },
            {
                "claim": "Electronic records maintained or submitted under FDA record requirements may be subject to 21 CFR Part 11 controls.",
                "title": "Part 11 Electronic Records and Electronic Signatures Scope and Application",
                "source_url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application",
                "source_type": "regulatory",
                "supports": ["electronic_records", "electronic_signatures", "audit_trail"],
            },
            {
                "claim": "Drug development tools are qualified only for a stated context of use, while product-specific evidence can be reviewed in an IND, NDA, or BLA.",
                "title": "FDA Drug Development Tool Qualification Programs",
                "source_url": "https://www.fda.gov/drugs/development-approval-process-drugs/drug-development-tool-ddt-qualification-programs",
                "source_type": "regulatory",
                "supports": ["regulatory_pathway", "context_of_use"],
            },
            {
                "claim": "CDER and CBER use eCTD as the standard electronic format for IND, NDA, and BLA submissions.",
                "title": "FDA Electronic Common Technical Document",
                "source_url": "https://www.fda.gov/drugs/electronic-regulatory-submission-and-review/electronic-common-technical-document-ectd",
                "source_type": "regulatory",
                "supports": ["ectd", "regulatory_export"],
            },
            {
                "claim": "Human intestinal immuno-organoids can reproduce clinically observed epithelial injury from a T-cell bispecific.",
                "title": "Human organoids with an autologous tissue-resident immune compartment",
                "source_url": "https://www.nature.com/articles/s41586-024-07791-5",
                "source_type": "peer_reviewed",
                "supports": ["immune_coculture", "biological_relevance"],
            },
            {
                "claim": "Bayesian active learning can prospectively reduce the fraction of a drug-combination space that must be measured.",
                "title": "A Bayesian active learning platform for scalable combination drug screens",
                "source_url": "https://www.nature.com/articles/s41467-024-55287-7",
                "source_type": "peer_reviewed",
                "supports": ["active_learning", "experiment_design"],
            },
        ]


class EvidenceAgent:
    name = "evidence_agent"

    def __init__(self, provider: EvidenceProvider | None = None) -> None:
        self.provider = provider or CuratedEvidenceProvider()

    def run(self, goal: ResearchGoal) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        for item in self.provider.collect(goal):
            claim = str(item["claim"])
            title = str(item["title"])
            source_url = str(item["source_url"])
            digest = hashlib.sha256(f"{claim}|{title}|{source_url}".encode()).hexdigest()
            records.append(
                EvidenceRecord(
                    project_id=goal.id,
                    claim=claim,
                    title=title,
                    source_url=source_url,
                    source_type=str(item["source_type"]),
                    supports=list(item.get("supports", [])),
                    content_hash=digest,
                )
            )
        return records


class ModelArchitectAgent:
    name = "model_architect_agent"

    def __init__(self, backend: ReasoningBackend | None = None) -> None:
        self.backend = backend or RuleBasedReasoningBackend()

    def run(self, goal: ResearchGoal, evidence: list[EvidenceRecord]) -> list[Hypothesis]:
        factor_names = sorted(goal.factor_space)
        evidence_ids = [record.id for record in evidence]
        hypotheses: list[Hypothesis] = []
        templates = [
            (
                "A bounded combination of approved culture factors can improve fidelity without reducing assay dynamic range.",
                "culture environment alters phenotype retention and functional responsiveness",
                0.90,
                0.55,
                0.15,
            ),
            (
                "A subset of approved factors drives reproducibility across biological replicates and batches.",
                "variance components are concentrated in controllable culture inputs",
                0.88,
                0.50,
                0.12,
            ),
            (
                "Cost-aware adaptive sampling can reach the qualification threshold with fewer conditions than a fixed design.",
                "posterior uncertainty identifies experiments with the highest expected information gain",
                0.95,
                0.62,
                0.10,
            ),
        ]
        for statement, mechanism, testability, novelty, risk in templates:
            hypotheses.append(
                Hypothesis(
                    project_id=goal.id,
                    statement=self.backend.summarize(statement, evidence),
                    mechanism=mechanism,
                    evidence_ids=evidence_ids,
                    testability_score=testability,
                    novelty_score=novelty,
                    risk_score=risk,
                    proposed_factors=factor_names,
                )
            )
        return hypotheses


class MetaReviewAgent:
    name = "meta_review_agent"

    def run(self, hypotheses: list[Hypothesis], limit: int = 2) -> list[Hypothesis]:
        return sorted(
            hypotheses,
            key=lambda item: (
                0.50 * item.testability_score
                + 0.25 * item.novelty_score
                + 0.25 * (1 - item.risk_score)
            ),
            reverse=True,
        )[:limit]


@dataclass(frozen=True)
class PlannedExperiment:
    plan: ExperimentPlan
    expected_information_gain: float


class ExperimentDesignAgent:
    name = "experiment_design_agent"

    def run(
        self,
        project: ProjectSnapshot,
        hypotheses: list[Hypothesis],
        history: list[tuple[object, float]],
        round_index: int,
    ) -> PlannedExperiment:
        space = ConditionSpace(project.goal)
        count = (
            project.goal.budget.initial_condition_limit
            if round_index == 1
            else project.goal.budget.subsequent_condition_limit
        )
        locked = project.goal.cro.study_phase is StudyPhase.LOCKED_VALIDATION
        if locked:
            count = project.goal.budget.subsequent_condition_limit
        if round_index == 1 or not history:
            design = space.initial_design(count)
        elif locked:
            unique: dict[str, object] = {}
            for condition, _ in history:
                unique.setdefault(condition.fingerprint(), condition)
            prior = list(unique.values())[:count]
            design = DesignResult(
                conditions=[
                    type(condition)(
                        factors=condition.factors,
                        estimated_cost=condition.estimated_cost,
                        rationale="locked prospective validation design",
                    )
                    for condition in prior
                ],
                expected_information_gain=1.0,
            )
        else:
            design = space.adaptive_design(count, history)  # type: ignore[arg-type]
        controls = [
            ControlSpec(
                control_type="negative",
                name="approved_negative_control",
                approved_material_id="control:negative:v1",
            ),
            ControlSpec(
                control_type="positive",
                name="approved_positive_control",
                approved_material_id="control:positive:v1",
            ),
            ControlSpec(
                control_type="mechanism",
                name="approved_mechanism_control",
                approved_material_id="control:mechanism:v1",
            ),
        ]
        estimated_cost = sum(item.estimated_cost for item in design.conditions)
        plan = ExperimentPlan(
            project_id=project.project_id,
            round_index=round_index,
            hypothesis_ids=[hypothesis.id for hypothesis in hypotheses],
            conditions=design.conditions,
            controls=controls,
            biological_replicates=3,
            technical_replicates=2,
            sop_version="SOP-DM-001-v1",
            protocol_id=project.goal.cro.protocol_id,
            protocol_version=project.goal.cro.protocol_version,
            sap_id=project.goal.cro.sap_id,
            randomization_seed=17 + round_index,
            blinded=project.goal.cro.blinded,
            design_locked=locked,
            acceptance_criteria={
                "maximum_within_plate_cv": project.goal.quality.maximum_within_plate_cv,
                "maximum_between_batch_cv": project.goal.quality.maximum_between_batch_cv,
                "minimum_z_prime": project.goal.quality.minimum_z_prime,
                "minimum_fidelity_score": project.goal.quality.minimum_fidelity_score,
                "minimum_driver_concordance": project.goal.quality.minimum_driver_concordance,
            },
            stop_rules=StopRules(
                maximum_rounds=project.goal.budget.maximum_rounds,
                information_gain_below=project.goal.quality.information_gain_stop_threshold,
                consecutive_quality_passes=project.goal.quality.required_consecutive_passes,
            ),
            estimated_cost=estimated_cost,
        )
        return PlannedExperiment(plan=plan, expected_information_gain=design.expected_information_gain)


class CriticComplianceAgent:
    name = "critic_compliance_agent"

    def __init__(self, policy: CompliancePolicy | None = None) -> None:
        self.policy = policy or CompliancePolicy()

    def review_goal(self, goal: ResearchGoal) -> PolicyReport:
        return self.policy.validate_goal(goal)

    def review_evidence(self, evidence: list[EvidenceRecord]) -> PolicyReport:
        return self.policy.validate_evidence(evidence)

    def review_plan(self, project: ProjectSnapshot, plan: ExperimentPlan) -> PolicyReport:
        remaining = project.goal.budget.maximum_cost - project.spent_cost
        return self.policy.validate_plan(project.goal, plan, remaining)

    def review_result(
        self, project: ProjectSnapshot, plan: ExperimentPlan, bundle: ResultBundle
    ) -> PolicyReport:
        findings: list[PolicyFinding] = []
        if bundle.site_id not in project.goal.governance.approved_sites:
            findings.append(
                PolicyFinding(
                    code="UNAPPROVED_SITE",
                    severity="critical",
                    message="result originates from an unapproved site",
                )
            )
        if bundle.project_id != project.project_id or bundle.plan_id != plan.id:
            findings.append(
                PolicyFinding(
                    code="RESULT_LINKAGE",
                    severity="critical",
                    message="result does not match the active project and plan",
                )
            )
        if (
            bundle.protocol_id != plan.protocol_id
            or bundle.protocol_version != plan.protocol_version
        ):
            findings.append(
                PolicyFinding(
                    code="PROTOCOL_MISMATCH",
                    severity="critical",
                    message="result protocol identity/version does not match the issued work order",
                )
            )
        if any(
            deviation.category == "critical"
            or deviation.disposition in {"open", "invalidates_result"}
            for deviation in bundle.deviations
        ):
            findings.append(
                PolicyFinding(
                    code="UNRESOLVED_DEVIATION",
                    severity="critical",
                    message="result contains a critical, open, or result-invalidating deviation",
                )
            )
        if (
            project.goal.regulatory.part11_required
            and not bundle.raw_artifacts
        ):
            findings.append(
                PolicyFinding(
                    code="RAW_DATA_REQUIRED",
                    severity="critical",
                    message="Part 11-scoped result bundles require checksum-addressed raw artifacts",
                )
            )
        allowed_ids = {condition.id for condition in plan.conditions} | {
            control.name for control in plan.controls
        }
        observed_ids = {observation.condition_id for observation in bundle.observations}
        unknown = observed_ids - allowed_ids
        if unknown:
            findings.append(
                PolicyFinding(
                    code="UNKNOWN_CONDITION",
                    severity="critical",
                    message=f"result contains unknown condition IDs: {sorted(unknown)}",
                )
            )
        missing = {condition.id for condition in plan.conditions} - observed_ids
        if missing:
            findings.append(
                PolicyFinding(
                    code="MISSING_CONDITION",
                    severity="critical",
                    message=f"result omits {len(missing)} planned conditions",
                )
            )
        replicate_ids = [observation.replicate_id for observation in bundle.observations]
        if len(replicate_ids) != len(set(replicate_ids)):
            findings.append(
                PolicyFinding(
                    code="DUPLICATE_REPLICATE_ID",
                    severity="critical",
                    message="replicate identifiers must be unique within a result bundle",
                )
            )
        expected_per_item = plan.biological_replicates * plan.technical_replicates
        counts: dict[str, int] = defaultdict(int)
        for observation in bundle.observations:
            counts[observation.condition_id] += 1
        incomplete = {
            item_id: counts.get(item_id, 0)
            for item_id in allowed_ids
            if counts.get(item_id, 0) != expected_per_item
        }
        if incomplete:
            findings.append(
                PolicyFinding(
                    code="REPLICATION_MISMATCH",
                    severity="critical",
                    message=(
                        f"planned items require exactly {expected_per_item} observations; "
                        f"mismatches: {incomplete}"
                    ),
                )
            )
        control_type_by_name = {control.name: control.control_type for control in plan.controls}
        mislabeled_controls = [
            observation.replicate_id
            for observation in bundle.observations
            if observation.condition_id in control_type_by_name
            and observation.control_type != control_type_by_name[observation.condition_id]
        ]
        if mislabeled_controls:
            findings.append(
                PolicyFinding(
                    code="CONTROL_LABEL_MISMATCH",
                    severity="critical",
                    message=f"control labels mismatch the work order: {mislabeled_controls[:5]}",
                )
            )
        return PolicyReport(
            passed=not any(item.severity == "critical" for item in findings), findings=findings
        )


class ProtocolCompilerAgent:
    name = "protocol_compiler_agent"

    rows = "ABCDEFGH"

    @classmethod
    def _well(cls, index: int) -> str:
        plate = index // 96 + 1
        local = index % 96
        row = cls.rows[local // 12]
        column = local % 12 + 1
        return f"P{plate}:{row}{column:02d}"

    def run(self, project: ProjectSnapshot, plan: ExperimentPlan) -> WorkOrder:
        assignments: list[PlateAssignment] = []
        index = 0
        for biological in range(1, plan.biological_replicates + 1):
            for technical in range(1, plan.technical_replicates + 1):
                for condition in plan.conditions:
                    assignments.append(
                        PlateAssignment(
                            well=self._well(index),
                            item_type="condition",
                            item_id=condition.id,
                            biological_replicate=biological,
                            technical_replicate=technical,
                        )
                    )
                    index += 1
                for control in plan.controls:
                    assignments.append(
                        PlateAssignment(
                            well=self._well(index),
                            item_type="control",
                            item_id=control.name,
                            biological_replicate=biological,
                            technical_replicate=technical,
                        )
                    )
                    index += 1
        return WorkOrder(
            project_id=project.project_id,
            plan_id=plan.id,
            site_id=project.goal.governance.approved_sites[0],
            sop_version=plan.sop_version,
            protocol_id=plan.protocol_id,
            protocol_version=plan.protocol_version,
            design_locked=plan.design_locked,
            assignments=assignments,
            instructions=[
                f"Execute approved {plan.sop_version}; this work order cannot amend the SOP.",
                "Use only the listed condition IDs and approved controls.",
                "Record sample, reagent, instrument and batch identifiers in the LIMS.",
                "Stop and escalate on contamination, identity mismatch, protocol deviation or access failure.",
                "Upload raw artifacts with SHA-256 checksums before submitting the ResultBundle.",
            ],
        )


class AnalysisAgent:
    name = "analysis_agent"

    @staticmethod
    def _cv(values: list[float]) -> float | None:
        if len(values) < 2:
            return None
        mean = statistics.fmean(values)
        if abs(mean) < 1e-12:
            return None
        return abs(statistics.stdev(values) / mean)

    def run(self, goal: ResearchGoal, bundle: ResultBundle) -> AnalysisReport:
        valid = [observation for observation in bundle.observations if observation.valid]
        grouped: dict[str, list[float]] = defaultdict(list)
        controls: dict[str, list[float]] = defaultdict(list)
        for observation in valid:
            grouped[observation.condition_id].append(observation.value)
            if observation.control_type:
                controls[observation.control_type].append(observation.value)
        condition_cvs = [
            cv for values in grouped.values() if (cv := self._cv(values)) is not None
        ]
        within_plate_cv = float(np.median(condition_cvs)) if condition_cvs else None
        z_prime: float | None = None
        if controls["positive"] and controls["negative"]:
            positive = controls["positive"]
            negative = controls["negative"]
            positive_sd = statistics.stdev(positive) if len(positive) > 1 else 0.0
            negative_sd = statistics.stdev(negative) if len(negative) > 1 else 0.0
            separation = abs(statistics.fmean(positive) - statistics.fmean(negative))
            if separation > 1e-12:
                z_prime = 1 - 3 * (positive_sd + negative_sd) / separation
        condition_scores = {
            condition_id: statistics.fmean(values)
            for condition_id, values in grouped.items()
            if condition_id not in {
                "approved_negative_control",
                "approved_positive_control",
                "approved_mechanism_control",
            }
        }
        invalid_rate = 1 - len(valid) / len(bundle.observations)
        qc = bundle.submitted_qc
        anomalies: list[str] = []
        if bundle.failed_reason:
            anomalies.append(bundle.failed_reason)
        if invalid_rate > 0.05:
            anomalies.append("invalid observation rate exceeds 5%")
        if within_plate_cv is None:
            anomalies.append("within-plate CV could not be calculated")
        if z_prime is None:
            anomalies.append("Z-prime could not be calculated")
        quality_passed = all(
            [
                not bundle.failed_reason,
                qc.mycoplasma_negative,
                qc.sample_identity_match,
                qc.driver_concordance >= goal.quality.minimum_driver_concordance,
                qc.fidelity_score >= goal.quality.minimum_fidelity_score,
                within_plate_cv is not None
                and within_plate_cv <= goal.quality.maximum_within_plate_cv,
                z_prime is not None and z_prime >= goal.quality.minimum_z_prime,
                qc.between_batch_cv is None
                or qc.between_batch_cv <= goal.quality.maximum_between_batch_cv,
                qc.ai_segmentation_dice is None
                or qc.ai_segmentation_dice >= goal.quality.minimum_site_ai_dice,
                invalid_rate <= 0.05,
            ]
        )
        return AnalysisReport(
            project_id=goal.id,
            plan_id=bundle.plan_id,
            result_id=bundle.id,
            within_plate_cv=within_plate_cv,
            z_prime=z_prime,
            between_batch_cv=qc.between_batch_cv,
            interlab_icc=qc.interlab_icc,
            ai_segmentation_dice=qc.ai_segmentation_dice,
            driver_concordance=qc.driver_concordance,
            fidelity_score=qc.fidelity_score,
            quality_passed=quality_passed,
            condition_scores=condition_scores,
            invalid_observation_rate=invalid_rate,
            anomalies=anomalies,
        )
