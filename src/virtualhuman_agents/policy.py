from __future__ import annotations

import re

from .models import (
    AIRegulatoryImpact,
    EvidenceRecord,
    ExperimentPlan,
    FactorKind,
    GLPApplicability,
    PolicyFinding,
    PolicyReport,
    RegulatoryPathway,
    ResearchGoal,
    StudyPhase,
)


INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore\s+(all|any|the)\s+(previous|prior)\s+instructions",
        r"system\s+prompt",
        r"developer\s+message",
        r"bypass\s+(safety|policy|approval)",
        r"reveal\s+(credentials|secrets|tokens)",
    ]
]


class CompliancePolicy:
    required_consent_scopes = {"organoid_research", "data_processing"}
    required_controls = {"negative", "positive", "mechanism"}

    def validate_goal(self, goal: ResearchGoal) -> PolicyReport:
        findings: list[PolicyFinding] = []
        governance = goal.governance
        if governance.data_residency != "CN":
            findings.append(
                PolicyFinding(
                    code="DATA_RESIDENCY",
                    severity="critical",
                    message="v1 requires China data residency",
                )
            )
        if governance.cross_border_transfer_allowed:
            findings.append(
                PolicyFinding(
                    code="CROSS_BORDER_DISABLED",
                    severity="critical",
                    message="cross-border transfer is outside the approved v1 envelope",
                )
            )
        missing = self.required_consent_scopes - set(governance.consent_scope)
        if missing:
            findings.append(
                PolicyFinding(
                    code="CONSENT_SCOPE",
                    severity="critical",
                    message=f"missing required consent scopes: {sorted(missing)}",
                )
            )
        if not governance.sample_reuse_allowed:
            findings.append(
                PolicyFinding(
                    code="SAMPLE_REUSE",
                    severity="warning",
                    message="sample reuse is disabled; cross-project model learning must remain off",
                )
            )
        if not governance.ethics_approval_id.strip():
            findings.append(
                PolicyFinding(
                    code="ETHICS_APPROVAL",
                    severity="critical",
                    message="ethics approval is required",
                )
            )
        cro = goal.cro
        if cro.study_director_user_id == cro.qau_user_id:
            findings.append(
                PolicyFinding(
                    code="QAU_INDEPENDENCE",
                    severity="critical",
                    message="study director and quality assurance unit reviewer must be independent",
                )
            )
        if cro.study_phase in {StudyPhase.VALIDATION, StudyPhase.LOCKED_VALIDATION}:
            if not cro.blinded or not cro.randomization_required:
                findings.append(
                    PolicyFinding(
                        code="VALIDATION_BIAS_CONTROL",
                        severity="critical",
                        message="validation studies require blinding and randomization",
                    )
                )
        regulatory = goal.regulatory
        if (
            regulatory.pathway is not RegulatoryPathway.RESEARCH_ONLY
            and not regulatory.early_fda_engagement_planned
        ):
            findings.append(
                PolicyFinding(
                    code="FDA_ENGAGEMENT",
                    severity="warning",
                    message="early FDA engagement is not planned for a regulatory-use model",
                )
            )
        if (
            regulatory.ai_regulatory_impact is AIRegulatoryImpact.SUPPORTS_REGULATORY_DECISION
            and regulatory.pathway is RegulatoryPathway.RESEARCH_ONLY
        ):
            findings.append(
                PolicyFinding(
                    code="AI_COU_PATHWAY",
                    severity="critical",
                    message="AI supporting a regulatory decision cannot use a research-only pathway",
                )
            )
        if (
            regulatory.glp_applicability is GLPApplicability.GLP
            and cro.study_phase not in {StudyPhase.VALIDATION, StudyPhase.LOCKED_VALIDATION}
        ):
            findings.append(
                PolicyFinding(
                    code="GLP_PHASE",
                    severity="critical",
                    message="a GLP claim requires a validation or locked-validation study phase",
                )
            )
        return PolicyReport(
            passed=not any(item.severity == "critical" for item in findings), findings=findings
        )

    def validate_evidence(self, evidence: list[EvidenceRecord]) -> PolicyReport:
        findings: list[PolicyFinding] = []
        if not evidence:
            findings.append(
                PolicyFinding(
                    code="NO_EVIDENCE", severity="critical", message="no traceable evidence supplied"
                )
            )
        for record in evidence:
            text = f"{record.claim}\n{record.title}"
            if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
                findings.append(
                    PolicyFinding(
                        code="PROMPT_INJECTION",
                        severity="critical",
                        message=f"evidence {record.id} contains instruction-like content",
                    )
                )
        return PolicyReport(
            passed=not any(item.severity == "critical" for item in findings), findings=findings
        )

    def validate_plan(
        self, goal: ResearchGoal, plan: ExperimentPlan, remaining_budget: float
    ) -> PolicyReport:
        findings: list[PolicyFinding] = []
        limit = (
            goal.budget.initial_condition_limit
            if plan.round_index == 1
            else goal.budget.subsequent_condition_limit
        )
        if len(plan.conditions) > limit:
            findings.append(
                PolicyFinding(
                    code="CONDITION_LIMIT",
                    severity="critical",
                    message=f"round {plan.round_index} exceeds the {limit}-condition limit",
                )
            )
        if plan.estimated_cost > remaining_budget:
            findings.append(
                PolicyFinding(
                    code="BUDGET_EXCEEDED",
                    severity="critical",
                    message="plan exceeds the remaining approved budget",
                )
            )
        control_types = {control.control_type for control in plan.controls}
        missing_controls = self.required_controls - control_types
        if missing_controls:
            findings.append(
                PolicyFinding(
                    code="MISSING_CONTROLS",
                    severity="critical",
                    message=f"missing controls: {sorted(missing_controls)}",
                )
            )
        if plan.biological_replicates < 3 or plan.technical_replicates < 2:
            findings.append(
                PolicyFinding(
                    code="INSUFFICIENT_REPLICATION",
                    severity="critical",
                    message="at least 3 biological and 2 technical replicates are required",
                )
            )
        if (
            plan.protocol_id != goal.cro.protocol_id
            or plan.protocol_version != goal.cro.protocol_version
            or plan.sap_id != goal.cro.sap_id
        ):
            findings.append(
                PolicyFinding(
                    code="CONTROLLED_DOCUMENT_MISMATCH",
                    severity="critical",
                    message="plan does not reference the approved protocol/SAP versions",
                )
            )
        if goal.cro.study_phase is StudyPhase.LOCKED_VALIDATION and not plan.design_locked:
            findings.append(
                PolicyFinding(
                    code="VALIDATION_NOT_LOCKED",
                    severity="critical",
                    message="locked validation prohibits adaptive changes to the design",
                )
            )
        if goal.cro.blinded and not plan.blinded:
            findings.append(
                PolicyFinding(
                    code="BLINDING_REMOVED",
                    severity="critical",
                    message="plan cannot remove protocol-required blinding",
                )
            )
        expected_criteria = {
            "maximum_within_plate_cv": goal.quality.maximum_within_plate_cv,
            "maximum_between_batch_cv": goal.quality.maximum_between_batch_cv,
            "minimum_z_prime": goal.quality.minimum_z_prime,
            "minimum_fidelity_score": goal.quality.minimum_fidelity_score,
            "minimum_driver_concordance": goal.quality.minimum_driver_concordance,
        }
        if plan.acceptance_criteria != expected_criteria:
            findings.append(
                PolicyFinding(
                    code="ACCEPTANCE_CRITERIA_DRIFT",
                    severity="critical",
                    message="plan acceptance criteria differ from the pre-registered goal",
                )
            )
        for condition in plan.conditions:
            unknown = set(condition.factors) - set(goal.factor_space)
            if unknown:
                findings.append(
                    PolicyFinding(
                        code="UNAPPROVED_FACTOR",
                        severity="critical",
                        message=f"condition {condition.id} uses unapproved factors: {sorted(unknown)}",
                    )
                )
                continue
            for name, value in condition.factors.items():
                spec = goal.factor_space[name]
                if spec.kind in {FactorKind.CONTINUOUS, FactorKind.INTEGER}:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        findings.append(
                            PolicyFinding(
                                code="FACTOR_TYPE",
                                severity="critical",
                                message=f"factor {name} requires a numeric value",
                            )
                        )
                    elif not float(spec.minimum) <= float(value) <= float(spec.maximum):
                        findings.append(
                            PolicyFinding(
                                code="FACTOR_RANGE",
                                severity="critical",
                                message=f"factor {name} is outside its approved range",
                            )
                        )
                    elif spec.kind is FactorKind.INTEGER and float(value) % 1:
                        findings.append(
                            PolicyFinding(
                                code="FACTOR_INTEGER",
                                severity="critical",
                                message=f"factor {name} must be an integer",
                            )
                        )
                elif value not in spec.choices:
                    findings.append(
                        PolicyFinding(
                            code="FACTOR_CHOICE",
                            severity="critical",
                            message=f"factor {name} uses an unapproved choice",
                        )
                    )
        return PolicyReport(
            passed=not any(item.severity == "critical" for item in findings), findings=findings
        )
