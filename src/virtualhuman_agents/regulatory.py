from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import (
    ApprovalRole,
    ControlledArtifact,
    ElectronicApproval,
    ProjectState,
    ReadinessCheck,
    RegulatoryPackage,
    RegulatoryPathway,
    RegulatoryReadinessReport,
    StudyPhase,
)
from .store import ConflictError, NotFoundError, SQLiteStore


class RegulatoryGateError(ConflictError):
    pass


def model_sha256(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class RegulatoryService:
    """CRO delivery and FDA-readiness controls; it never claims FDA acceptance or approval."""

    required_artifact_types = {
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
    }

    def __init__(self, store: SQLiteStore, package_root: str | Path = "regulatory_packages") -> None:
        self.store = store
        self.package_root = Path(package_root)

    def register_artifact(self, artifact: ControlledArtifact) -> ControlledArtifact:
        self.store.get_project(artifact.project_id)
        self.store.add_controlled_artifact(artifact)
        self.store.append_audit(
            artifact.project_id,
            artifact.owner_user_id,
            "controlled_artifact.registered",
            {
                "artifact_id": artifact.id,
                "artifact_type": artifact.artifact_type,
                "version": artifact.version,
                "sha256": artifact.sha256,
            },
        )
        return artifact

    def object_hash(self, project_id: str, object_type: str, object_id: str) -> str:
        project = self.store.get_project(project_id)
        if object_type == "goal":
            if object_id != project.goal.id:
                raise NotFoundError(object_id)
            return model_sha256(project.goal)
        collections: dict[str, list[BaseModel]] = {
            "plan": list(self.store.list_plans(project_id)),
            "result": list(self.store.list_results(project_id)),
            "analysis": list(self.store.list_analyses(project_id)),
            "controlled_artifact": list(self.store.list_controlled_artifacts(project_id)),
            "regulatory_package": list(self.store.list_regulatory_packages(project_id)),
        }
        if object_type not in collections:
            raise ValueError(f"unsupported approval object type: {object_type}")
        for item in collections[object_type]:
            if item.id == object_id:  # type: ignore[attr-defined]
                if object_type == "regulatory_package":
                    return item.sha256  # type: ignore[attr-defined, no-any-return]
                return model_sha256(item)
        raise NotFoundError(object_id)

    def approve(self, approval: ElectronicApproval) -> ElectronicApproval:
        project = self.store.get_project(approval.project_id)
        expected_hash = self.object_hash(
            approval.project_id, approval.object_type, approval.object_id
        )
        if approval.object_sha256.lower() != expected_hash.lower():
            raise ConflictError("approval hash does not match the immutable object")
        expected_signers = {
            ApprovalRole.SPONSOR: project.goal.cro.sponsor_representative_user_id,
            ApprovalRole.STUDY_DIRECTOR: project.goal.cro.study_director_user_id,
            ApprovalRole.QUALITY_ASSURANCE: project.goal.cro.qau_user_id,
        }
        expected_signer = expected_signers.get(approval.signer_role)
        if expected_signer and approval.signer_user_id != expected_signer:
            raise ConflictError("signer is not assigned to the declared CRO study role")
        duplicate = any(
            item.object_type == approval.object_type
            and item.object_id == approval.object_id
            and item.signer_role == approval.signer_role
            and item.meaning == approval.meaning
            for item in self.store.list_approvals(approval.project_id)
        )
        if duplicate:
            raise ConflictError("the same role has already signed this object with that meaning")
        self.store.add_approval(approval)
        self.store.append_audit(
            approval.project_id,
            approval.signer_user_id,
            "electronic_approval.recorded",
            {
                "approval_id": approval.id,
                "object_type": approval.object_type,
                "object_id": approval.object_id,
                "object_sha256": approval.object_sha256,
                "role": approval.signer_role.value,
                "meaning": approval.meaning,
                "authentication_event_id": approval.authentication_event_id,
            },
        )
        return approval

    @staticmethod
    def _check(code: str, passed: bool, evidence: str, critical: bool = True) -> ReadinessCheck:
        return ReadinessCheck(code=code, passed=passed, critical=critical, evidence=evidence)

    def assess_readiness(self, project_id: str, *, persist: bool = True) -> RegulatoryReadinessReport:
        project = self.store.get_project(project_id)
        goal = project.goal
        plans = self.store.list_plans(project_id)
        results = self.store.list_results(project_id)
        analyses = self.store.list_analyses(project_id)
        evidence = self.store.list_evidence(project_id)
        artifacts = [
            item for item in self.store.list_controlled_artifacts(project_id) if item.status == "effective"
        ]
        approvals = self.store.list_approvals(project_id)
        artifact_types = {item.artifact_type for item in artifacts}
        result_sites = {item.site_id for item in results}
        sponsor_goal_approved = any(
            item.object_type == "goal"
            and item.object_id == goal.id
            and item.signer_role is ApprovalRole.SPONSOR
            and item.meaning == "approved"
            for item in approvals
        )
        all_plans_approved = bool(plans) and all(
            any(
                approval.object_type == "plan"
                and approval.object_id == plan.id
                and approval.signer_role is ApprovalRole.STUDY_DIRECTOR
                and approval.meaning == "approved"
                for approval in approvals
            )
            for plan in plans
        )
        all_analyses_reviewed = bool(analyses) and all(
            any(
                approval.object_type == "analysis"
                and approval.object_id == analysis.id
                and approval.signer_role is ApprovalRole.QUALITY_ASSURANCE
                and approval.meaning in {"reviewed", "approved"}
                for approval in approvals
            )
            for analysis in analyses
        )
        open_deviations = [
            deviation
            for result in results
            for deviation in result.deviations
            if deviation.disposition in {"open", "invalidates_result"}
            or deviation.category == "critical"
        ]
        checks = [
            self._check(
                "REGULATORY_PATHWAY",
                goal.regulatory.pathway is not RegulatoryPathway.RESEARCH_ONLY,
                f"declared pathway: {goal.regulatory.pathway.value}",
            ),
            self._check(
                "CONTEXT_OF_USE",
                all(
                    len(value.strip()) >= 5
                    for value in [
                        goal.context_of_use,
                        goal.regulatory.question_of_interest,
                        goal.regulatory.model_role,
                        goal.regulatory.consequence_of_error,
                    ]
                ),
                "COU includes question of interest, model role, and consequence of error",
            ),
            self._check(
                "PROJECT_COMPLETED",
                project.state is ProjectState.COMPLETED,
                f"project state: {project.state.value}",
            ),
            self._check(
                "LOCKED_PROSPECTIVE_VALIDATION",
                goal.cro.study_phase is StudyPhase.LOCKED_VALIDATION
                and bool(plans)
                and all(plan.design_locked for plan in plans),
                f"study phase: {goal.cro.study_phase.value}; plans: {len(plans)}",
            ),
            self._check(
                "CONTROLLED_DOCUMENTS",
                self.required_artifact_types <= artifact_types,
                "missing: " + ", ".join(sorted(self.required_artifact_types - artifact_types)),
            ),
            self._check(
                "PROTOCOL_AND_SAP_TRACEABILITY",
                bool(plans)
                and all(
                    plan.protocol_id == goal.cro.protocol_id
                    and plan.protocol_version == goal.cro.protocol_version
                    and plan.sap_id == goal.cro.sap_id
                    for plan in plans
                ),
                f"protocol {goal.cro.protocol_id}/{goal.cro.protocol_version}; SAP {goal.cro.sap_id}",
            ),
            self._check(
                "QUALITY_ACCEPTANCE",
                len(analyses) >= goal.quality.required_consecutive_passes
                and all(report.quality_passed for report in analyses[-goal.quality.required_consecutive_passes :]),
                f"passing analyses: {sum(report.quality_passed for report in analyses)}/{len(analyses)}",
            ),
            self._check(
                "INDEPENDENT_SITE_REPLICATION",
                len(result_sites) >= 2
                and any(
                    report.interlab_icc is not None
                    and report.interlab_icc >= goal.quality.minimum_interlab_icc
                    for report in analyses
                ),
                f"sites: {sorted(result_sites)}; minimum ICC: {goal.quality.minimum_interlab_icc}",
            ),
            self._check(
                "RAW_DATA_INTEGRITY",
                bool(results)
                and all(result.raw_artifacts for result in results),
                f"checksum-addressed result bundles: {sum(bool(r.raw_artifacts) for r in results)}/{len(results)}",
            ),
            self._check(
                "DEVIATIONS_RESOLVED",
                not open_deviations,
                f"unresolved/critical deviations: {len(open_deviations)}",
            ),
            self._check(
                "TRACEABLE_EVIDENCE",
                len(evidence) >= 3 and sum(record.source_type == "regulatory" for record in evidence) >= 3,
                f"evidence records: {len(evidence)}; regulatory sources: {sum(r.source_type == 'regulatory' for r in evidence)}",
            ),
            self._check(
                "ELECTRONIC_APPROVALS",
                sponsor_goal_approved and all_plans_approved and all_analyses_reviewed,
                (
                    f"sponsor goal approval: {sponsor_goal_approved}; "
                    f"all plans approved: {all_plans_approved}; "
                    f"all analyses QAU-reviewed: {all_analyses_reviewed}"
                ),
            ),
            self._check(
                "TAMPER_EVIDENT_AUDIT_TRAIL",
                self.store.verify_audit_chain(project_id),
                f"audit events: {len(self.store.list_audit(project_id))}",
            ),
        ]
        ready = all(check.passed for check in checks if check.critical)
        limitations = [
            "This report demonstrates submission readiness, not FDA acceptance, qualification, or product approval.",
            "The generated archive is Module 4/DDT staging content, not a valid eCTD sequence.",
            "Part 11 and GLP compliance also require validated infrastructure, SOPs, training, and operational controls outside this software.",
        ]
        report = RegulatoryReadinessReport(
            project_id=project_id,
            pathway=goal.regulatory.pathway,
            checks=checks,
            ready_for_qa_release=ready,
            limitations=limitations,
        )
        if persist:
            self.store.add_readiness_report(report)
            self.store.append_audit(
                project_id,
                "regulatory_agent",
                "regulatory_readiness.assessed",
                {
                    "report_id": report.id,
                    "ready": ready,
                    "failed_checks": [check.code for check in checks if check.critical and not check.passed],
                },
            )
        return report

    def build_package(self, project_id: str) -> RegulatoryPackage:
        report = self.assess_readiness(project_id)
        if not report.ready_for_qa_release:
            failed = [check.code for check in report.checks if check.critical and not check.passed]
            raise RegulatoryGateError(f"regulatory readiness failed: {', '.join(failed)}")
        project = self.store.get_project(project_id)
        prefix = (
            "ddt-qualification"
            if project.goal.regulatory.pathway is RegulatoryPathway.DDT_QUALIFICATION
            else "m4/4.2.1-pharmacology"
        )
        payloads: dict[str, Any] = {
            f"{prefix}/context-of-use.json": {
                "context_of_use": project.goal.context_of_use,
                "question_of_interest": project.goal.regulatory.question_of_interest,
                "model_role": project.goal.regulatory.model_role,
                "consequence_of_error": project.goal.regulatory.consequence_of_error,
                "claims": project.goal.regulatory.regulatory_claims,
            },
            f"{prefix}/study-report.json": {
                "goal": project.goal.model_dump(mode="json"),
                "plans": [item.model_dump(mode="json") for item in self.store.list_plans(project_id)],
                "analyses": [
                    item.model_dump(mode="json") for item in self.store.list_analyses(project_id)
                ],
            },
            f"{prefix}/data/observations.json": [
                item.model_dump(mode="json") for item in self.store.list_results(project_id)
            ],
            f"{prefix}/data/reviewers-guide.json": {
                "standard": project.goal.regulatory.study_data_standard,
                "traceability": "ResultBundle -> ExperimentPlan -> ResearchGoal",
                "standardization_note": "Organoid/NAM endpoints may require a sponsor-agreed custom representation; confirm current FDA Data Standards Catalog before submission.",
            },
            f"{prefix}/methods/controlled-artifacts.json": [
                item.model_dump(mode="json")
                for item in self.store.list_controlled_artifacts(project_id)
            ],
            f"{prefix}/evidence.json": [
                item.model_dump(mode="json") for item in self.store.list_evidence(project_id)
            ],
            f"{prefix}/audit-trail.json": [
                item.model_dump(mode="json") for item in self.store.list_audit(project_id)
            ],
            f"{prefix}/approvals.json": [
                item.model_dump(mode="json") for item in self.store.list_approvals(project_id)
            ],
            "regulatory-readiness.json": report.model_dump(mode="json"),
            "PACKAGE-NOTICE.json": {
                "status": "staging_content_only",
                "notice": "Not an FDA submission and not evidence of FDA approval or DDT qualification.",
                "publishing": f"Publish and validate with the sponsor's validated eCTD {project.goal.regulatory.ectd_version} system where applicable.",
            },
        }
        encoded = {
            name: json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode()
            for name, value in payloads.items()
        }
        manifest = {
            "project_id": project_id,
            "files": [
                {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
                for name, content in sorted(encoded.items())
            ],
        }
        encoded["manifest.json"] = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2
        ).encode()
        package_id = f"regpkg_{hashlib.sha256(report.id.encode()).hexdigest()[:24]}"
        output_dir = self.package_root / project_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{package_id}.zip"
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(encoded.items()):
                archive.writestr(name, content)
        package_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
        package = RegulatoryPackage(
            id=package_id,
            project_id=project_id,
            readiness_report_id=report.id,
            package_format=(
                "DDT_QUALIFICATION_STAGING"
                if project.goal.regulatory.pathway is RegulatoryPathway.DDT_QUALIFICATION
                else "ECTD_MODULE4_STAGING"
            ),
            artifact_uri=str(output_path.resolve()),
            sha256=package_sha,
            status="ready_for_qa",
        )
        self.store.add_regulatory_package(package)
        self.store.append_audit(
            project_id,
            "regulatory_agent",
            "regulatory_package.generated",
            {"package_id": package.id, "sha256": package.sha256, "format": package.package_format},
        )
        return package

    def qa_release(self, project_id: str, package_id: str) -> RegulatoryPackage:
        package = self.store.get_regulatory_package(package_id)
        if package.project_id != project_id:
            raise NotFoundError(package_id)
        matching = {
            approval.signer_role
            for approval in self.store.list_approvals(project_id)
            if approval.object_type == "regulatory_package"
            and approval.object_id == package_id
            and approval.object_sha256 == package.sha256
            and approval.meaning in {"approved", "qa_released"}
        }
        required = {ApprovalRole.SPONSOR, ApprovalRole.QUALITY_ASSURANCE}
        if not required <= matching:
            raise RegulatoryGateError("sponsor and QAU package approvals are required for release")
        released = package.model_copy(update={"status": "qa_released"})
        self.store.update_regulatory_package(released)
        self.store.append_audit(
            project_id,
            "regulatory_agent",
            "regulatory_package.qa_released",
            {"package_id": package_id, "sha256": package.sha256},
        )
        return released
