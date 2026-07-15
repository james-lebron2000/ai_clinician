from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel

from .models import (
    AgentDecision,
    AnalysisReport,
    AuditEvent,
    ControlledArtifact,
    ElectronicApproval,
    EvidenceRecord,
    ExperimentPlan,
    Hypothesis,
    ProjectSnapshot,
    ProjectState,
    ResearchGoal,
    RegulatoryPackage,
    RegulatoryReadinessReport,
    ResultBundle,
    WorkOrder,
    utcnow,
)

T = TypeVar("T", bound=BaseModel)


class ConflictError(RuntimeError):
    pass


class NotFoundError(KeyError):
    pass


class SQLiteStore:
    """Small, append-oriented store with an immutable audit trail."""

    def __init__(self, path: str | Path = "virtualhuman.db") -> None:
        self.path = str(path)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    goal_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    spent_cost REAL NOT NULL DEFAULT 0,
                    consecutive_quality_passes INTEGER NOT NULL DEFAULT 0,
                    active_plan_id TEXT,
                    active_work_order_id TEXT,
                    last_decision_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id),
                    UNIQUE(project_id, round_index)
                );
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS work_orders (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    project_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE INDEX IF NOT EXISTS idx_audit_project
                    ON audit_events(project_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_plans_project
                    ON plans(project_id, round_index);
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS readiness_reports (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS regulatory_packages (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE TABLE IF NOT EXISTS controlled_artifacts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                """
            )
            # Non-destructive migration for databases created by the early prototype.
            audit_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(audit_events)")
            }
            if "previous_hash" not in audit_columns:
                connection.execute(
                    "ALTER TABLE audit_events ADD COLUMN previous_hash TEXT NOT NULL DEFAULT ''"
                )
            if "event_hash" not in audit_columns:
                connection.execute(
                    "ALTER TABLE audit_events ADD COLUMN event_hash TEXT NOT NULL DEFAULT ''"
                )

    @staticmethod
    def _dump(model: BaseModel) -> str:
        return model.model_dump_json()

    @staticmethod
    def _load(model_type: type[T], payload: str) -> T:
        return model_type.model_validate_json(payload)

    def create_project(self, goal: ResearchGoal) -> ProjectSnapshot:
        now = utcnow()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO projects (
                    project_id, goal_json, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (goal.id, self._dump(goal), ProjectState.DRAFT.value, now.isoformat(), now.isoformat()),
            )
        self.append_audit(goal.id, "pi_orchestrator", "project.created", {"goal_id": goal.id})
        return self.get_project(goal.id)

    def get_project(self, project_id: str) -> ProjectSnapshot:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(project_id)
        decision = (
            AgentDecision.model_validate_json(row["last_decision_json"])
            if row["last_decision_json"]
            else None
        )
        return ProjectSnapshot(
            project_id=row["project_id"],
            goal=ResearchGoal.model_validate_json(row["goal_json"]),
            state=ProjectState(row["state"]),
            spent_cost=row["spent_cost"],
            consecutive_quality_passes=row["consecutive_quality_passes"],
            active_plan_id=row["active_plan_id"],
            active_work_order_id=row["active_work_order_id"],
            last_decision=decision,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_project(
        self,
        project_id: str,
        *,
        expected_state: ProjectState | None = None,
        state: ProjectState | None = None,
        spent_cost: float | None = None,
        consecutive_quality_passes: int | None = None,
        active_plan_id: str | None | object = ...,
        active_work_order_id: str | None | object = ...,
        last_decision: AgentDecision | None | object = ...,
    ) -> ProjectSnapshot:
        current = self.get_project(project_id)
        if expected_state is not None and current.state is not expected_state:
            raise ConflictError(
                f"expected project state {expected_state.value}, got {current.state.value}"
            )
        values: dict[str, Any] = {"updated_at": utcnow().isoformat()}
        if state is not None:
            values["state"] = state.value
        if spent_cost is not None:
            values["spent_cost"] = spent_cost
        if consecutive_quality_passes is not None:
            values["consecutive_quality_passes"] = consecutive_quality_passes
        if active_plan_id is not ...:
            values["active_plan_id"] = active_plan_id
        if active_work_order_id is not ...:
            values["active_work_order_id"] = active_work_order_id
        if last_decision is not ...:
            values["last_decision_json"] = (
                self._dump(last_decision) if isinstance(last_decision, AgentDecision) else None
            )
        assignments = ", ".join(f"{key} = ?" for key in values)
        parameters = [*values.values(), project_id]
        if expected_state is not None:
            query = f"UPDATE projects SET {assignments} WHERE project_id = ? AND state = ?"
            parameters.append(expected_state.value)
        else:
            query = f"UPDATE projects SET {assignments} WHERE project_id = ?"
        with self.connection() as connection:
            cursor = connection.execute(query, parameters)
            if cursor.rowcount != 1:
                raise ConflictError("project was concurrently modified")
        return self.get_project(project_id)

    def _insert_model(self, table: str, model: BaseModel, **columns: Any) -> None:
        keys = ["id", "project_id", *columns, "payload"]
        values = [getattr(model, "id"), getattr(model, "project_id"), *columns.values(), self._dump(model)]
        placeholders = ", ".join("?" for _ in keys)
        with self.connection() as connection:
            try:
                connection.execute(
                    f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({placeholders})", values
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictError(str(exc)) from exc

    def add_evidence(self, record: EvidenceRecord) -> None:
        self._insert_model("evidence", record)

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        self._insert_model("hypotheses", hypothesis)

    def add_plan(self, plan: ExperimentPlan) -> None:
        self._insert_model("plans", plan, round_index=plan.round_index)

    def add_result(self, result: ResultBundle) -> None:
        self._insert_model("results", result, plan_id=result.plan_id)

    def add_analysis(self, report: AnalysisReport) -> None:
        self._insert_model("analyses", report, plan_id=report.plan_id)

    def add_decision(self, decision: AgentDecision) -> None:
        self._insert_model("decisions", decision)

    def add_work_order(self, order: WorkOrder) -> None:
        self._insert_model("work_orders", order, plan_id=order.plan_id)

    def add_approval(self, approval: ElectronicApproval) -> None:
        self._insert_model("approvals", approval)

    def add_readiness_report(self, report: RegulatoryReadinessReport) -> None:
        self._insert_model("readiness_reports", report)

    def add_regulatory_package(self, package: RegulatoryPackage) -> None:
        self._insert_model("regulatory_packages", package)

    def update_regulatory_package(self, package: RegulatoryPackage) -> None:
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE regulatory_packages SET payload = ? WHERE id = ?",
                (self._dump(package), package.id),
            )
            if cursor.rowcount != 1:
                raise NotFoundError(package.id)

    def add_controlled_artifact(self, artifact: ControlledArtifact) -> None:
        self._insert_model("controlled_artifacts", artifact)

    def _list_models(
        self,
        table: str,
        model_type: type[T],
        project_id: str,
        order_by: str = "rowid",
    ) -> list[T]:
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT payload FROM {table} WHERE project_id = ? ORDER BY {order_by}",
                (project_id,),
            ).fetchall()
        return [self._load(model_type, row["payload"]) for row in rows]

    def list_evidence(self, project_id: str) -> list[EvidenceRecord]:
        return self._list_models("evidence", EvidenceRecord, project_id)

    def list_hypotheses(self, project_id: str) -> list[Hypothesis]:
        return self._list_models("hypotheses", Hypothesis, project_id)

    def list_plans(self, project_id: str) -> list[ExperimentPlan]:
        return self._list_models("plans", ExperimentPlan, project_id, "round_index")

    def list_results(self, project_id: str) -> list[ResultBundle]:
        return self._list_models("results", ResultBundle, project_id)

    def list_analyses(self, project_id: str) -> list[AnalysisReport]:
        return self._list_models("analyses", AnalysisReport, project_id)

    def list_decisions(self, project_id: str) -> list[AgentDecision]:
        return self._list_models("decisions", AgentDecision, project_id)

    def list_work_orders(self, project_id: str) -> list[WorkOrder]:
        return self._list_models("work_orders", WorkOrder, project_id)

    def list_approvals(self, project_id: str) -> list[ElectronicApproval]:
        return self._list_models("approvals", ElectronicApproval, project_id)

    def list_readiness_reports(self, project_id: str) -> list[RegulatoryReadinessReport]:
        return self._list_models("readiness_reports", RegulatoryReadinessReport, project_id)

    def list_regulatory_packages(self, project_id: str) -> list[RegulatoryPackage]:
        return self._list_models("regulatory_packages", RegulatoryPackage, project_id)

    def list_controlled_artifacts(self, project_id: str) -> list[ControlledArtifact]:
        return self._list_models("controlled_artifacts", ControlledArtifact, project_id)

    def get_plan(self, plan_id: str) -> ExperimentPlan:
        return self._get_by_id("plans", ExperimentPlan, plan_id)

    def get_work_order(self, work_order_id: str) -> WorkOrder:
        return self._get_by_id("work_orders", WorkOrder, work_order_id)

    def get_result(self, result_id: str) -> ResultBundle:
        return self._get_by_id("results", ResultBundle, result_id)

    def get_analysis(self, analysis_id: str) -> AnalysisReport:
        return self._get_by_id("analyses", AnalysisReport, analysis_id)

    def get_regulatory_package(self, package_id: str) -> RegulatoryPackage:
        return self._get_by_id("regulatory_packages", RegulatoryPackage, package_id)

    def get_controlled_artifact(self, artifact_id: str) -> ControlledArtifact:
        return self._get_by_id("controlled_artifacts", ControlledArtifact, artifact_id)

    def _get_by_id(self, table: str, model_type: type[T], item_id: str) -> T:
        with self.connection() as connection:
            row = connection.execute(
                f"SELECT payload FROM {table} WHERE id = ?", (item_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(item_id)
        return self._load(model_type, row["payload"])

    def append_audit(
        self, project_id: str, actor: str, action: str, payload: dict[str, Any] | None = None
    ) -> AuditEvent:
        with self.connection() as connection:
            prior = connection.execute(
                "SELECT event_hash FROM audit_events WHERE project_id = ? ORDER BY sequence DESC LIMIT 1",
                (project_id,),
            ).fetchone()
            previous_hash = prior["event_hash"] if prior and prior["event_hash"] else "0" * 64
            event_id = f"audit_{uuid.uuid4().hex}"
            created_at = utcnow()
            normalized_payload = payload or {}
            payload_json = json.dumps(
                normalized_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
            )
            material = "|".join(
                [previous_hash, event_id, project_id, actor, action, payload_json, created_at.isoformat()]
            )
            event_hash = hashlib.sha256(material.encode()).hexdigest()
            event = AuditEvent(
                id=event_id,
                project_id=project_id,
                actor=actor,
                action=action,
                payload=normalized_payload,
                previous_hash=previous_hash,
                event_hash=event_hash,
                created_at=created_at,
            )
            connection.execute(
                """
                INSERT INTO audit_events (
                    id, project_id, actor, action, payload, previous_hash, event_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.project_id,
                    event.actor,
                    event.action,
                    payload_json,
                    event.previous_hash,
                    event.event_hash,
                    event.created_at.isoformat(),
                ),
            )
        return event

    def list_audit(self, project_id: str) -> list[AuditEvent]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, project_id, actor, action, payload, previous_hash, event_hash, created_at
                FROM audit_events WHERE project_id = ? ORDER BY sequence
                """,
                (project_id,),
            ).fetchall()
        return [
            AuditEvent(
                id=row["id"],
                project_id=row["project_id"],
                actor=row["actor"],
                action=row["action"],
                payload=json.loads(row["payload"]),
                previous_hash=row["previous_hash"],
                event_hash=row["event_hash"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def verify_audit_chain(self, project_id: str) -> bool:
        previous_hash = "0" * 64
        for event in self.list_audit(project_id):
            if event.previous_hash != previous_hash:
                return False
            payload_json = json.dumps(
                event.payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            material = "|".join(
                [
                    event.previous_hash,
                    event.id,
                    event.project_id,
                    event.actor,
                    event.action,
                    payload_json,
                    event.created_at.isoformat(),
                ]
            )
            if hashlib.sha256(material.encode()).hexdigest() != event.event_hash:
                return False
            previous_hash = event.event_hash
        return True
