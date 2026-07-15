from __future__ import annotations

from .agents import (
    AnalysisAgent,
    CriticComplianceAgent,
    EvidenceAgent,
    ExperimentDesignAgent,
    MetaReviewAgent,
    ModelArchitectAgent,
    ProtocolCompilerAgent,
)
from .models import (
    AgentDecision,
    CycleOutcome,
    DecisionAction,
    EvidenceRecord,
    ExperimentalCondition,
    ExperimentPlan,
    ProjectSnapshot,
    ProjectState,
    ResearchGoal,
    ResultBundle,
    WorkOrder,
)
from .store import ConflictError, SQLiteStore


class PIOrchestrator:
    """Stateful principal-investigator agent for the bounded autonomous loop."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store
        self.evidence_agent = EvidenceAgent()
        self.architect_agent = ModelArchitectAgent()
        self.design_agent = ExperimentDesignAgent()
        self.protocol_agent = ProtocolCompilerAgent()
        self.analysis_agent = AnalysisAgent()
        self.critic_agent = CriticComplianceAgent()
        self.meta_review_agent = MetaReviewAgent()

    def create_project(self, goal: ResearchGoal) -> ProjectSnapshot:
        project = self.store.create_project(goal)
        report = self.critic_agent.review_goal(goal)
        self.store.append_audit(
            goal.id,
            self.critic_agent.name,
            "goal.reviewed",
            report.model_dump(mode="json"),
        )
        if report.passed:
            project = self.store.update_project(
                goal.id, expected_state=ProjectState.DRAFT, state=ProjectState.READY
            )
        else:
            decision = AgentDecision(
                project_id=goal.id,
                action=DecisionAction.PAUSE,
                rationale="Governance gate failed before autonomous execution.",
                requires_human_approval=True,
                approval_reasons=[finding.message for finding in report.findings],
            )
            self.store.add_decision(decision)
            project = self.store.update_project(
                goal.id,
                expected_state=ProjectState.DRAFT,
                state=ProjectState.PAUSED,
                last_decision=decision,
            )
        return project

    def start(
        self, project_id: str, evidence_records: list[EvidenceRecord] | None = None
    ) -> CycleOutcome:
        project = self.store.get_project(project_id)
        if project.state is not ProjectState.READY:
            raise ConflictError(f"project must be ready, got {project.state.value}")
        self.store.update_project(
            project_id,
            expected_state=ProjectState.READY,
            state=ProjectState.EVIDENCE_REVIEW,
        )
        evidence = evidence_records or self.evidence_agent.run(project.goal)
        for record in evidence:
            if record.project_id != project_id:
                raise ConflictError("evidence project_id does not match the active project")
            self.store.add_evidence(record)
        evidence_report = self.critic_agent.review_evidence(evidence)
        self.store.append_audit(
            project_id,
            self.evidence_agent.name,
            "evidence.collected",
            {"count": len(evidence), "record_ids": [record.id for record in evidence]},
        )
        self.store.append_audit(
            project_id,
            self.critic_agent.name,
            "evidence.reviewed",
            evidence_report.model_dump(mode="json"),
        )
        if not evidence_report.passed:
            return self._pause(
                project_id,
                "Evidence safety or traceability gate failed.",
                [finding.message for finding in evidence_report.findings],
                expected_state=ProjectState.EVIDENCE_REVIEW,
            )

        self.store.update_project(
            project_id,
            expected_state=ProjectState.EVIDENCE_REVIEW,
            state=ProjectState.DESIGNING,
        )
        hypotheses = self.architect_agent.run(project.goal, evidence)
        for hypothesis in hypotheses:
            self.store.add_hypothesis(hypothesis)
        selected = self.meta_review_agent.run(hypotheses)
        self.store.append_audit(
            project_id,
            self.meta_review_agent.name,
            "hypotheses.selected",
            {
                "candidate_count": len(hypotheses),
                "selected_ids": [hypothesis.id for hypothesis in selected],
            },
        )
        refreshed = self.store.get_project(project_id)
        planned = self.design_agent.run(refreshed, selected, [], round_index=1)
        return self._review_compile_issue(refreshed, planned.plan, planned.expected_information_gain)

    def submit_results(self, project_id: str, bundle: ResultBundle) -> CycleOutcome:
        project = self.store.get_project(project_id)
        if project.state is not ProjectState.AWAITING_RESULTS:
            raise ConflictError(f"project is not awaiting results: {project.state.value}")
        if project.active_plan_id is None:
            raise ConflictError("project has no active plan")
        plan = self.store.get_plan(project.active_plan_id)
        result_report = self.critic_agent.review_result(project, plan, bundle)
        self.store.append_audit(
            project_id,
            self.critic_agent.name,
            "result.reviewed",
            result_report.model_dump(mode="json"),
        )
        if not result_report.passed:
            return self._pause(
                project_id,
                "Result linkage or applicability-domain gate failed.",
                [finding.message for finding in result_report.findings],
                expected_state=ProjectState.AWAITING_RESULTS,
            )
        self.store.add_result(bundle)
        self.store.update_project(
            project_id,
            expected_state=ProjectState.AWAITING_RESULTS,
            state=ProjectState.ANALYZING,
            active_work_order_id=None,
        )
        analysis = self.analysis_agent.run(project.goal, bundle)
        self.store.add_analysis(analysis)
        self.store.append_audit(
            project_id,
            self.analysis_agent.name,
            "result.analyzed",
            analysis.model_dump(mode="json"),
        )
        self.store.update_project(
            project_id,
            expected_state=ProjectState.ANALYZING,
            state=ProjectState.DECIDING,
        )

        passes = project.consecutive_quality_passes + 1 if analysis.quality_passed else 0
        self.store.update_project(
            project_id,
            expected_state=ProjectState.DECIDING,
            consecutive_quality_passes=passes,
        )
        if not analysis.quality_passed:
            outcome = self._pause(
                project_id,
                "Quality gate failed; autonomous iteration is halted.",
                analysis.anomalies or ["one or more predefined quality thresholds failed"],
                expected_state=ProjectState.DECIDING,
            )
            return outcome.model_copy(update={"analysis": analysis})
        if passes >= project.goal.quality.required_consecutive_passes:
            decision = AgentDecision(
                project_id=project_id,
                action=DecisionAction.COMPLETE,
                rationale="The predefined quality threshold passed in consecutive independent rounds.",
                evidence_ids=[item.id for item in self.store.list_evidence(project_id)],
                expected_information_gain=0.0,
            )
            self.store.add_decision(decision)
            completed = self.store.update_project(
                project_id,
                expected_state=ProjectState.DECIDING,
                state=ProjectState.COMPLETED,
                active_plan_id=None,
                last_decision=decision,
            )
            self.store.append_audit(
                project_id, self.meta_review_agent.name, "project.completed", {"decision_id": decision.id}
            )
            return CycleOutcome(project=completed, decision=decision, analysis=analysis)

        plans = self.store.list_plans(project_id)
        next_round = len(plans) + 1
        if next_round > project.goal.budget.maximum_rounds:
            decision = AgentDecision(
                project_id=project_id,
                action=DecisionAction.COMPLETE,
                rationale="Maximum approved rounds reached; stopped without consecutive qualification.",
                requires_human_approval=True,
                approval_reasons=["qualification target was not confirmed twice"],
            )
            self.store.add_decision(decision)
            completed = self.store.update_project(
                project_id,
                expected_state=ProjectState.DECIDING,
                state=ProjectState.COMPLETED,
                active_plan_id=None,
                last_decision=decision,
            )
            return CycleOutcome(project=completed, decision=decision, analysis=analysis)

        history = self._history(project_id)
        selected = self.meta_review_agent.run(self.store.list_hypotheses(project_id))
        refreshed = self.store.get_project(project_id)
        planned = self.design_agent.run(refreshed, selected, history, round_index=next_round)
        if planned.expected_information_gain < project.goal.quality.information_gain_stop_threshold:
            decision = AgentDecision(
                project_id=project_id,
                action=DecisionAction.COMPLETE,
                rationale="Expected information gain fell below the pre-registered stop threshold.",
                expected_information_gain=planned.expected_information_gain,
                requires_human_approval=True,
                approval_reasons=["stopped before consecutive qualification"],
            )
            self.store.add_decision(decision)
            completed = self.store.update_project(
                project_id,
                expected_state=ProjectState.DECIDING,
                state=ProjectState.COMPLETED,
                active_plan_id=None,
                last_decision=decision,
            )
            return CycleOutcome(project=completed, decision=decision, analysis=analysis)
        outcome = self._review_compile_issue(
            self.store.get_project(project_id),
            planned.plan,
            planned.expected_information_gain,
            expected_state=ProjectState.DECIDING,
            decision_action=DecisionAction.NEXT_ROUND,
        )
        return outcome.model_copy(update={"analysis": analysis})

    def resume(self, project_id: str, approval_note: str) -> CycleOutcome:
        project = self.store.get_project(project_id)
        if project.state is not ProjectState.PAUSED:
            raise ConflictError("only paused projects can be resumed")
        if len(approval_note.strip()) < 10:
            raise ValueError("a substantive corrective-action note is required")
        plans = self.store.list_plans(project_id)
        if not plans:
            raise ConflictError("governance failures require a corrected new project definition")
        next_round = len(plans) + 1
        if next_round > project.goal.budget.maximum_rounds:
            raise ConflictError("maximum approved rounds have been consumed")
        self.store.append_audit(
            project_id,
            "human_research_lead",
            "project.resume_approved",
            {"corrective_action": approval_note},
        )
        self.store.update_project(
            project_id, expected_state=ProjectState.PAUSED, state=ProjectState.DESIGNING
        )
        selected = self.meta_review_agent.run(self.store.list_hypotheses(project_id))
        planned = self.design_agent.run(
            self.store.get_project(project_id),
            selected,
            self._history(project_id),
            round_index=next_round,
        )
        return self._review_compile_issue(
            self.store.get_project(project_id),
            planned.plan,
            planned.expected_information_gain,
            expected_state=ProjectState.DESIGNING,
            decision_action=DecisionAction.NEXT_ROUND,
        )

    def _history(self, project_id: str) -> list[tuple[ExperimentalCondition, float]]:
        plan_by_id = {plan.id: plan for plan in self.store.list_plans(project_id)}
        history: list[tuple[ExperimentalCondition, float]] = []
        for analysis in self.store.list_analyses(project_id):
            plan = plan_by_id[analysis.plan_id]
            condition_by_id = {condition.id: condition for condition in plan.conditions}
            for condition_id, score in analysis.condition_scores.items():
                if condition_id in condition_by_id:
                    history.append((condition_by_id[condition_id], score))
        return history

    def _review_compile_issue(
        self,
        project: ProjectSnapshot,
        plan: ExperimentPlan,
        expected_information_gain: float,
        *,
        expected_state: ProjectState = ProjectState.DESIGNING,
        decision_action: DecisionAction = DecisionAction.ISSUE_WORK_ORDER,
    ) -> CycleOutcome:
        policy_report = self.critic_agent.review_plan(project, plan)
        self.store.append_audit(
            project.project_id,
            self.critic_agent.name,
            "plan.reviewed",
            policy_report.model_dump(mode="json"),
        )
        if not policy_report.passed:
            return self._pause(
                project.project_id,
                "Experiment plan exceeded the approved autonomous envelope.",
                [finding.message for finding in policy_report.findings],
                expected_state=expected_state,
            )
        self.store.add_plan(plan)
        order = self.protocol_agent.run(project, plan)
        self.store.add_work_order(order)
        decision = AgentDecision(
            project_id=project.project_id,
            action=decision_action,
            rationale="Plan passed compliance review and was compiled into an immutable LIMS work order.",
            evidence_ids=[record.id for record in self.store.list_evidence(project.project_id)],
            expected_information_gain=expected_information_gain,
            estimated_cost=plan.estimated_cost,
            uncertainty=min(1.0, expected_information_gain),
            next_plan_id=plan.id,
        )
        self.store.add_decision(decision)
        updated = self.store.update_project(
            project.project_id,
            expected_state=expected_state,
            state=ProjectState.AWAITING_RESULTS,
            spent_cost=project.spent_cost + plan.estimated_cost,
            active_plan_id=plan.id,
            active_work_order_id=order.id,
            last_decision=decision,
        )
        self.store.append_audit(
            project.project_id,
            self.protocol_agent.name,
            "work_order.issued",
            {
                "work_order_id": order.id,
                "plan_id": plan.id,
                "assignment_count": len(order.assignments),
            },
        )
        return CycleOutcome(project=updated, decision=decision, work_order=order)

    def _pause(
        self,
        project_id: str,
        rationale: str,
        reasons: list[str],
        *,
        expected_state: ProjectState,
    ) -> CycleOutcome:
        decision = AgentDecision(
            project_id=project_id,
            action=DecisionAction.PAUSE,
            rationale=rationale,
            requires_human_approval=True,
            approval_reasons=reasons,
        )
        self.store.add_decision(decision)
        project = self.store.update_project(
            project_id,
            expected_state=expected_state,
            state=ProjectState.PAUSED,
            active_work_order_id=None,
            last_decision=decision,
        )
        self.store.append_audit(
            project_id,
            self.critic_agent.name,
            "project.paused",
            {"decision_id": decision.id, "reasons": reasons},
        )
        return CycleOutcome(project=project, decision=decision)
