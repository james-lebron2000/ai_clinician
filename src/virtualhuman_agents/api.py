from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai_clinician.privacy import is_placeholder_secret

from .models import (
    ControlledArtifact,
    CycleOutcome,
    ElectronicApproval,
    ProjectSnapshot,
    RegulatoryPackage,
    RegulatoryReadinessReport,
    ResearchGoal,
    ResultBundle,
)
from .orchestrator import PIOrchestrator
from .regulatory import RegulatoryGateError, RegulatoryService
from .store import ConflictError, NotFoundError, SQLiteStore


class ResumeRequest(BaseModel):
    approval_note: str = Field(min_length=10)


def _authorizer(expected_key: str):
    def authorize(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key is None or not hmac.compare_digest(x_api_key, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid research API key required",
            )

    return authorize


def create_app(
    db_path: str | Path | None = None,
    package_root: str | Path | None = None,
    *,
    api_key: str | None = None,
) -> FastAPI:
    expected_key = api_key or os.getenv("VIRTUALHUMAN_API_KEY")
    if expected_key is None or len(expected_key.encode("utf-8")) < 16:
        raise RuntimeError("VIRTUALHUMAN_API_KEY must contain at least 16 bytes")
    if is_placeholder_secret(expected_key):
        raise RuntimeError("VIRTUALHUMAN_API_KEY cannot be a placeholder")
    store = SQLiteStore(db_path or os.getenv("VIRTUALHUMAN_DB", "virtualhuman.db"))
    orchestrator = PIOrchestrator(store)
    regulatory = RegulatoryService(
        store,
        package_root or os.getenv("VIRTUALHUMAN_PACKAGE_ROOT", "regulatory_packages"),
    )
    app = FastAPI(
        title="VirtualHuman Disease Model R&D Agent API",
        version="0.1.0",
        description=(
            "Bounded autonomous disease-model R&D with CRO governance and FDA-readiness gates. "
            "Readiness is not FDA approval."
        ),
        dependencies=[Depends(_authorizer(expected_key))],
    )
    app.state.store = store
    app.state.orchestrator = orchestrator
    app.state.regulatory = regulatory

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"not found: {exc.args[0]}"})

    @app.exception_handler(ConflictError)
    async def conflict(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "virtualhuman-agents"}

    @app.post("/projects", response_model=ProjectSnapshot, status_code=201)
    def create_project(goal: ResearchGoal) -> ProjectSnapshot:
        return orchestrator.create_project(goal)

    @app.get("/projects/{project_id}", response_model=ProjectSnapshot)
    def get_project(project_id: str) -> ProjectSnapshot:
        return store.get_project(project_id)

    @app.post("/projects/{project_id}/start", response_model=CycleOutcome)
    def start_project(project_id: str) -> CycleOutcome:
        return orchestrator.start(project_id)

    @app.post("/projects/{project_id}/results", response_model=CycleOutcome)
    def submit_results(project_id: str, bundle: ResultBundle) -> CycleOutcome:
        return orchestrator.submit_results(project_id, bundle)

    @app.post("/projects/{project_id}/resume", response_model=CycleOutcome)
    def resume_project(project_id: str, request: ResumeRequest) -> CycleOutcome:
        return orchestrator.resume(project_id, request.approval_note)

    @app.get("/projects/{project_id}/plans")
    def plans(project_id: str):
        store.get_project(project_id)
        return store.list_plans(project_id)

    @app.get("/projects/{project_id}/work-orders")
    def work_orders(project_id: str):
        store.get_project(project_id)
        return store.list_work_orders(project_id)

    @app.get("/projects/{project_id}/analyses")
    def analyses(project_id: str):
        store.get_project(project_id)
        return store.list_analyses(project_id)

    @app.get("/projects/{project_id}/audit")
    def audit(project_id: str):
        store.get_project(project_id)
        return {
            "chain_valid": store.verify_audit_chain(project_id),
            "events": store.list_audit(project_id),
        }

    @app.post("/projects/{project_id}/controlled-artifacts", response_model=ControlledArtifact)
    def register_artifact(project_id: str, artifact: ControlledArtifact) -> ControlledArtifact:
        if artifact.project_id != project_id:
            raise ValueError("artifact project_id does not match URL")
        return regulatory.register_artifact(artifact)

    @app.get("/projects/{project_id}/approval-hash/{object_type}/{object_id}")
    def approval_hash(project_id: str, object_type: str, object_id: str) -> dict[str, str]:
        return {"sha256": regulatory.object_hash(project_id, object_type, object_id)}

    @app.post("/projects/{project_id}/approvals", response_model=ElectronicApproval)
    def approve(project_id: str, approval: ElectronicApproval) -> ElectronicApproval:
        if approval.project_id != project_id:
            raise ValueError("approval project_id does not match URL")
        return regulatory.approve(approval)

    @app.post(
        "/projects/{project_id}/regulatory-readiness",
        response_model=RegulatoryReadinessReport,
    )
    def assess_readiness(project_id: str) -> RegulatoryReadinessReport:
        return regulatory.assess_readiness(project_id)

    @app.post(
        "/projects/{project_id}/regulatory-packages",
        response_model=RegulatoryPackage,
        status_code=201,
    )
    def build_package(project_id: str) -> RegulatoryPackage:
        return regulatory.build_package(project_id)

    @app.post(
        "/projects/{project_id}/regulatory-packages/{package_id}/qa-release",
        response_model=RegulatoryPackage,
    )
    def qa_release(project_id: str, package_id: str) -> RegulatoryPackage:
        return regulatory.qa_release(project_id, package_id)

    return app
