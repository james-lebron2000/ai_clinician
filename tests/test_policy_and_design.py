from __future__ import annotations

from virtualhuman_agents.agents import ExperimentDesignAgent
from virtualhuman_agents.demo import immune_coculture_goal
from virtualhuman_agents.models import (
    ExperimentalCondition,
    ExperimentPlan,
    ProjectState,
)
from virtualhuman_agents.orchestrator import PIOrchestrator
from virtualhuman_agents.policy import CompliancePolicy


def test_cro_role_independence_is_mandatory(goal):
    invalid = goal.model_copy(
        update={
            "cro": goal.cro.model_copy(
                update={"qau_user_id": goal.cro.study_director_user_id}
            )
        }
    )
    report = CompliancePolicy().validate_goal(invalid)
    assert not report.passed
    assert "QAU_INDEPENDENCE" in {item.code for item in report.findings}


def test_initial_locked_design_uses_approved_envelope(store, goal):
    orchestrator = PIOrchestrator(store)
    project = orchestrator.create_project(goal)
    outcome = orchestrator.start(project.project_id)
    assert outcome.project.state is ProjectState.AWAITING_RESULTS
    plan = store.get_plan(outcome.project.active_plan_id)
    assert plan.design_locked is True
    assert len(plan.conditions) == goal.budget.subsequent_condition_limit
    assert {control.control_type for control in plan.controls} == {
        "negative",
        "positive",
        "mechanism",
    }
    assert plan.protocol_id == goal.cro.protocol_id
    assert plan.sap_id == goal.cro.sap_id


def test_out_of_range_factor_is_rejected(store, goal):
    orchestrator = PIOrchestrator(store)
    project = orchestrator.create_project(goal)
    outcome = orchestrator.start(project.project_id)
    original = store.get_plan(outcome.project.active_plan_id)
    bad_condition = original.conditions[0].model_copy(
        update={"factors": {**original.conditions[0].factors, "egf_ng_ml": 999.0}}
    )
    bad_plan = original.model_copy(
        update={"id": "plan-redteam", "conditions": [bad_condition, *original.conditions[1:]]}
    )
    report = CompliancePolicy().validate_plan(goal, bad_plan, goal.budget.maximum_cost)
    assert not report.passed
    assert "FACTOR_RANGE" in {item.code for item in report.findings}


def test_second_sentinel_migrates_without_agent_changes(store):
    goal = immune_coculture_goal()
    outcome = PIOrchestrator(store).start(
        PIOrchestrator(store).create_project(goal).project_id
    )
    assert outcome.project.state is ProjectState.AWAITING_RESULTS
    plan = store.get_plan(outcome.project.active_plan_id)
    assert len(plan.conditions) == 24
    assert "immune_epithelial_ratio" in plan.conditions[0].factors
