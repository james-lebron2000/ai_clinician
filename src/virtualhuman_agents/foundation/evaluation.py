from __future__ import annotations

from .schemas import (
    BenchmarkResult,
    EvaluationGate,
    FoundationReadinessReport,
    HumanFMConfig,
    MetricDirection,
    RegistryAuditReport,
)


def assess_foundation_readiness(
    config: HumanFMConfig,
    registry_audit: RegistryAuditReport,
    results: list[BenchmarkResult],
    gates: list[EvaluationGate],
) -> FoundationReadinessReport:
    by_key = {(result.metric, result.subgroup): result for result in results}
    failed: list[str] = []
    for gate in gates:
        result = by_key.get((gate.metric, gate.subgroup))
        label = f"{gate.metric}[{gate.subgroup}]"
        if result is None:
            if gate.critical:
                failed.append(f"{label}:missing")
            continue
        passed = (
            result.value >= gate.threshold
            if gate.direction is MetricDirection.GREATER_OR_EQUAL
            else result.value <= gate.threshold
        )
        if not passed and gate.critical:
            failed.append(f"{label}:{result.value}")

    warnings: list[str] = []
    result_names = {result.metric for result in results}
    for required in [
        "ood_recall",
        "calibration_error",
        "external_site_performance",
        "perturbation_response_error",
    ]:
        if required not in result_names:
            warnings.append(f"recommended evaluation missing: {required}")
    if not registry_audit.passed:
        failed.append("data_registry_audit")
    return FoundationReadinessReport(
        model_name=config.name,
        model_version=config.version,
        context_of_use=config.context_of_use,
        registry_audit_passed=registry_audit.passed,
        passed=not failed,
        failed_gates=failed,
        warnings=warnings,
    )
