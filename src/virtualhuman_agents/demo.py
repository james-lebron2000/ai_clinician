from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from .models import (
    AIRegulatoryImpact,
    ApprovalRole,
    ArtifactRef,
    Budget,
    CROStudyGovernance,
    ControlledArtifact,
    ElectronicApproval,
    FactorKind,
    FactorSpec,
    GLPApplicability,
    Governance,
    Observation,
    ProjectState,
    RegulatoryPathway,
    RegulatoryProfile,
    ResearchGoal,
    ResultBundle,
    StudyPhase,
    SubmittedQC,
)
from .orchestrator import PIOrchestrator
from .regulatory import RegulatoryService
from .store import SQLiteStore


def sentinel_goal(*, locked_validation: bool = True) -> ResearchGoal:
    return ResearchGoal(
        title="AI-agent optimized colorectal cancer patient-derived organoid model",
        disease="colorectal cancer",
        therapeutic_context="oncology pharmacology and translational response testing",
        decision_context="select a reproducible, biologically faithful model for sponsor drug development",
        context_of_use=(
            "A patient-derived colorectal tumor organoid assay used as supportive nonclinical "
            "evidence to rank candidate treatment conditions in a product-specific IND program."
        ),
        target_phenotypes=["tissue fidelity", "drug-response dynamic range", "reproducibility"],
        primary_readout="normalized viability response",
        factor_space={
            "egf_ng_ml": FactorSpec(
                kind=FactorKind.CONTINUOUS, unit="ng/mL", minimum=10, maximum=50
            ),
            "matrix_percent": FactorSpec(
                kind=FactorKind.CONTINUOUS, unit="%", minimum=60, maximum=90
            ),
            "seeding_cells": FactorSpec(
                kind=FactorKind.INTEGER, unit="cells/well", minimum=500, maximum=1500
            ),
            "medium_variant": FactorSpec(
                kind=FactorKind.CATEGORICAL, choices=["approved_A", "approved_B"]
            ),
        },
        budget=Budget(
            maximum_cost=480000,
            maximum_rounds=4,
            initial_condition_limit=24,
            subsequent_condition_limit=16,
        ),
        governance=Governance(
            ethics_approval_id="IRB-CRC-2026-001",
            consent_scope=["organoid_research", "data_processing", "drug_development"],
            sample_reuse_allowed=True,
            cross_project_training_allowed=False,
            data_residency="CN",
            cross_border_transfer_allowed=False,
            approved_sites=["CRO-SHANGHAI-01", "CRO-SUZHOU-02"],
        ),
        cro=CROStudyGovernance(
            sponsor_id="SPONSOR-DEMO",
            cro_id="CRO-DEMO",
            study_id="VH-CRC-001",
            protocol_id="VH-CRC-PROT-001",
            protocol_version="2.0",
            sap_id="VH-CRC-SAP-001",
            study_phase=(StudyPhase.LOCKED_VALIDATION if locked_validation else StudyPhase.DEVELOPMENT),
            study_director_user_id="user-study-director",
            qau_user_id="user-independent-qau",
            sponsor_representative_user_id="user-sponsor-rep",
            blinded=True,
            randomization_required=True,
            vendor_qualification_ids=["VQ-MATRIX-001", "VQ-SEQUENCING-001"],
        ),
        regulatory=RegulatoryProfile(
            pathway=RegulatoryPathway.PRODUCT_SPECIFIC_SUBMISSION,
            glp_applicability=(
                GLPApplicability.HYBRID if locked_validation else GLPApplicability.NON_GLP_EXPLORATORY
            ),
            part11_required=True,
            fda_application_type="IND",
            ectd_version="4.0",
            study_data_standard="CUSTOM_WITH_REVIEWERS_GUIDE",
            ai_regulatory_impact=AIRegulatoryImpact.SUPPORTS_INTERPRETATION,
            question_of_interest="Which approved model condition provides a fit-for-purpose drug-response assay?",
            model_role="supportive nonclinical evidence; not a stand-alone safety or efficacy decision",
            consequence_of_error="incorrect candidate prioritization and additional confirmatory studies",
            early_fda_engagement_planned=True,
            regulatory_claims=[
                "technical reproducibility within the stated assay envelope",
                "biological fidelity to the source tumor within the stated donor population",
            ],
        ),
    )


def immune_coculture_goal() -> ResearchGoal:
    """Second sentinel: paired normal intestinal organoid–immune co-culture."""

    return ResearchGoal(
        title="Paired normal intestinal organoid and immune-cell co-culture model",
        disease="immune-mediated intestinal injury",
        therapeutic_context="immunotoxicity and mechanism-of-action assessment",
        decision_context="select a transferable co-culture configuration with barrier and immune function",
        context_of_use=(
            "A paired donor intestinal organoid–immune co-culture used during exploratory drug "
            "development to compare barrier injury and immune activation across approved conditions."
        ),
        target_phenotypes=["epithelial barrier integrity", "immune activation", "cell composition"],
        primary_readout="composite barrier and cytokine response",
        factor_space={
            "immune_epithelial_ratio": FactorSpec(
                kind=FactorKind.CONTINUOUS, minimum=0.1, maximum=1.0
            ),
            "co_culture_hours": FactorSpec(
                kind=FactorKind.INTEGER, unit="h", minimum=24, maximum=96
            ),
            "stimulation": FactorSpec(
                kind=FactorKind.CATEGORICAL,
                choices=["approved_baseline", "approved_stimulus_low", "approved_stimulus_high"],
            ),
            "donor_pairing": FactorSpec(kind=FactorKind.BOOLEAN),
        },
        budget=Budget(
            maximum_cost=720000,
            maximum_rounds=4,
            initial_condition_limit=24,
            subsequent_condition_limit=16,
        ),
        governance=Governance(
            ethics_approval_id="IRB-IMMUNE-GUT-2026-001",
            consent_scope=["organoid_research", "data_processing", "immune_coculture"],
            sample_reuse_allowed=True,
            cross_project_training_allowed=False,
            data_residency="CN",
            cross_border_transfer_allowed=False,
            approved_sites=["CRO-SHANGHAI-01", "CRO-SUZHOU-02"],
        ),
        cro=CROStudyGovernance(
            sponsor_id="SPONSOR-DEMO",
            cro_id="CRO-DEMO",
            study_id="VH-GUT-IMMUNE-001",
            protocol_id="VH-GUT-IMMUNE-PROT-001",
            protocol_version="1.0",
            sap_id="VH-GUT-IMMUNE-SAP-001",
            study_phase=StudyPhase.DEVELOPMENT,
            study_director_user_id="user-study-director",
            qau_user_id="user-independent-qau",
            sponsor_representative_user_id="user-sponsor-rep",
            blinded=True,
            randomization_required=True,
            vendor_qualification_ids=["VQ-IMMUNE-CELLS-001", "VQ-MATRIX-001"],
        ),
        regulatory=RegulatoryProfile(
            pathway=RegulatoryPathway.PRODUCT_SPECIFIC_SUBMISSION,
            glp_applicability=GLPApplicability.NON_GLP_EXPLORATORY,
            part11_required=True,
            fda_application_type="IND",
            ectd_version="4.0",
            study_data_standard="CUSTOM_WITH_REVIEWERS_GUIDE",
            ai_regulatory_impact=AIRegulatoryImpact.SUPPORTS_INTERPRETATION,
            question_of_interest="Which co-culture configuration is fit for exploratory immunotoxicity ranking?",
            model_role="exploratory supportive evidence that requires later locked validation",
            consequence_of_error="incorrect prioritization of a co-culture condition",
            early_fda_engagement_planned=True,
            regulatory_claims=["transferability of the control strategy to a complex immune model"],
        ),
    )


def _numeric_score(factors: dict[str, str | float | int | bool]) -> float:
    egf = float(factors["egf_ng_ml"])
    matrix = float(factors["matrix_percent"])
    seeding = float(factors["seeding_cells"])
    medium = 0.05 if factors["medium_variant"] == "approved_B" else 0.0
    response = (
        0.72
        - 0.00045 * (egf - 32) ** 2
        - 0.00035 * (matrix - 76) ** 2
        - 0.00000025 * (seeding - 1000) ** 2
        + medium
    )
    return max(0.20, min(0.95, response))


def synthetic_result(
    store: SQLiteStore,
    project_id: str,
    *,
    site_id: str | None = None,
) -> ResultBundle:
    project = store.get_project(project_id)
    if not project.active_plan_id:
        raise RuntimeError("project has no active plan")
    plan = store.get_plan(project.active_plan_id)
    round_index = plan.round_index
    selected_site = site_id or project.goal.governance.approved_sites[(round_index - 1) % 2]
    observations: list[Observation] = []
    for condition_index, condition in enumerate(plan.conditions):
        base = _numeric_score(condition.factors)
        for biological in range(1, plan.biological_replicates + 1):
            for technical in range(1, plan.technical_replicates + 1):
                noise = ((biological - 2) * 0.004) + ((technical - 1.5) * 0.003)
                observations.append(
                    Observation(
                        condition_id=condition.id,
                        replicate_id=f"R{round_index}-C{condition_index}-B{biological}-T{technical}",
                        biological_replicate=f"DONOR-{biological:02d}",
                        technical_replicate=technical,
                        value=base + noise,
                        metrics={"fidelity": 0.88 + 0.01 * math.sin(condition_index)},
                    )
                )
    control_values = {
        "negative": ("approved_negative_control", 0.10),
        "positive": ("approved_positive_control", 1.00),
        "mechanism": ("approved_mechanism_control", 0.55),
    }
    for control_type, (condition_id, base) in control_values.items():
        for biological in range(1, 4):
            for technical in range(1, 3):
                observations.append(
                    Observation(
                        condition_id=condition_id,
                        replicate_id=f"R{round_index}-{control_type}-B{biological}-T{technical}",
                        biological_replicate=f"CONTROL-{biological}",
                        technical_replicate=technical,
                        value=base + (biological - 2) * 0.002 + (technical - 1.5) * 0.001,
                        control_type=control_type,
                    )
                )
    raw_identity = f"{project_id}|{plan.id}|{selected_site}|batch-{round_index}"
    return ResultBundle(
        project_id=project_id,
        plan_id=plan.id,
        site_id=selected_site,
        batch_id=f"BATCH-{round_index:02d}",
        protocol_id=plan.protocol_id,
        protocol_version=plan.protocol_version,
        analyst_user_id=f"analyst-{selected_site.lower()}",
        instrument_ids=[f"IMAGER-{selected_site}-01"],
        reagent_lot_ids=[f"MATRIX-LOT-{round_index:02d}", "MEDIA-LOT-VALIDATED"],
        observations=observations,
        submitted_qc=SubmittedQC(
            mycoplasma_negative=True,
            sample_identity_match=True,
            driver_concordance=0.96,
            fidelity_score=0.88,
            between_batch_cv=0.11,
            interlab_icc=0.82 if round_index >= 2 else None,
            ai_segmentation_dice=0.92,
        ),
        raw_artifacts=[
            ArtifactRef(
                uri=f"lims://{selected_site}/{project_id}/{plan.id}/raw-images",
                sha256=hashlib.sha256(raw_identity.encode()).hexdigest(),
                media_type="application/vnd.ome-tiff",
            )
        ],
    )


def _register_demo_artifacts(service: RegulatoryService, project_id: str) -> None:
    project = service.store.get_project(project_id)
    for artifact_type in sorted(service.required_artifact_types):
        identity = f"{project_id}|{artifact_type}|v1"
        service.register_artifact(
            ControlledArtifact(
                project_id=project_id,
                artifact_type=artifact_type,
                title=f"Controlled {artifact_type.replace('_', ' ')} for {project.goal.cro.study_id}",
                version="1.0",
                uri=f"dms://validated/{project_id}/{artifact_type}/1.0",
                sha256=hashlib.sha256(identity.encode()).hexdigest(),
                owner_user_id="user-document-control",
            )
        )


def _approve(
    service: RegulatoryService,
    project_id: str,
    object_type: str,
    object_id: str,
    role: ApprovalRole,
    signer: str,
    meaning: str = "approved",
) -> ElectronicApproval:
    return service.approve(
        ElectronicApproval(
            project_id=project_id,
            object_type=object_type,
            object_id=object_id,
            object_sha256=service.object_hash(project_id, object_type, object_id),
            signer_user_id=signer,
            signer_role=role,
            meaning=meaning,
            authentication_event_id=f"idp-reauth-{hashlib.sha256((object_id + signer).encode()).hexdigest()[:16]}",
        )
    )


def run_demo(
    database_path: str | Path = "virtualhuman-demo.db",
    package_root: str | Path = "regulatory_packages",
) -> dict[str, Any]:
    store = SQLiteStore(database_path)
    orchestrator = PIOrchestrator(store)
    regulatory = RegulatoryService(store, package_root)
    project = orchestrator.create_project(sentinel_goal(locked_validation=True))
    outcome = orchestrator.start(project.project_id)
    while outcome.project.state is ProjectState.AWAITING_RESULTS:
        outcome = orchestrator.submit_results(
            project.project_id, synthetic_result(store, project.project_id)
        )
    if outcome.project.state is not ProjectState.COMPLETED:
        raise RuntimeError(f"demo did not complete: {outcome.project.state.value}")
    _register_demo_artifacts(regulatory, project.project_id)
    _approve(
        regulatory,
        project.project_id,
        "goal",
        project.goal.id,
        ApprovalRole.SPONSOR,
        project.goal.cro.sponsor_representative_user_id,
    )
    for plan in store.list_plans(project.project_id):
        _approve(
            regulatory,
            project.project_id,
            "plan",
            plan.id,
            ApprovalRole.STUDY_DIRECTOR,
            project.goal.cro.study_director_user_id,
        )
    for analysis in store.list_analyses(project.project_id):
        _approve(
            regulatory,
            project.project_id,
            "analysis",
            analysis.id,
            ApprovalRole.QUALITY_ASSURANCE,
            project.goal.cro.qau_user_id,
            "reviewed",
        )
    readiness = regulatory.assess_readiness(project.project_id)
    package = regulatory.build_package(project.project_id)
    _approve(
        regulatory,
        project.project_id,
        "regulatory_package",
        package.id,
        ApprovalRole.QUALITY_ASSURANCE,
        project.goal.cro.qau_user_id,
        "qa_released",
    )
    _approve(
        regulatory,
        project.project_id,
        "regulatory_package",
        package.id,
        ApprovalRole.SPONSOR,
        project.goal.cro.sponsor_representative_user_id,
    )
    released = regulatory.qa_release(project.project_id, package.id)
    return {
        "project_id": project.project_id,
        "state": store.get_project(project.project_id).state.value,
        "rounds": len(store.list_plans(project.project_id)),
        "result_sites": sorted({item.site_id for item in store.list_results(project.project_id)}),
        "audit_chain_valid": store.verify_audit_chain(project.project_id),
        "regulatory_ready": readiness.ready_for_qa_release,
        "package_status": released.status,
        "package_path": released.artifact_uri,
        "package_sha256": released.sha256,
    }
