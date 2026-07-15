"""Pre-specified, research-only target-trial emulation utilities.

This module estimates associations under explicit causal assumptions.  It does
not establish clinical efficacy, select a single best treatment, or create an
order.  Positivity, sample-size, balance, OOD, and uncertainty failures produce
an explicit abstention instead of a treatment claim.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from datetime import date, datetime, time, timezone

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ai_clinician.models import (
    ApplicabilityReport,
    CausalAuditArtifact,
    CausalEffectEstimate,
    TargetTrialInputSnapshot,
    TargetTrialSpec,
    TargetTrialResult,
    TreatmentOption,
    new_id,
)


RESEARCH_ONLY_DISCLAIMER = (
    "Single-centre observational research estimate; candidate strategy only. "
    "It does not establish clinical efficacy and must not be used to place orders."
)


class CausalAnalysisError(ValueError):
    """Raised when a target trial or its analytic dataset is malformed."""


def _read(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _identifier(value: Any) -> str:
    candidate = _read(value, "id", "treatment_id", "strategy_id", "label")
    if candidate is None:
        candidate = value
    return str(candidate).strip()


def _as_utc_timepoint(value: Any) -> datetime:
    """Parse provenance timestamps without accepting ambiguous numeric epochs."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    elif isinstance(value, str) and value.strip():
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = f"{candidate[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise CausalAnalysisError("time-zero provenance must use ISO-8601") from exc
    else:
        raise CausalAnalysisError("time-zero provenance must use ISO-8601")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_target_trial_spec(spec: TargetTrialSpec | Mapping[str, Any]) -> None:
    """Validate that a protocol is pre-specified enough to be estimable."""

    errors: list[str] = []
    if not str(_read(spec, "name", default="")).strip():
        errors.append("name")
    if not str(
        _read(spec, "decision_point_kind", "decision_point", default="")
    ).strip():
        errors.append("decision_point_kind")
    eligibility = _read(spec, "eligibility_criteria", "eligibility", default=None)
    if eligibility in (None, {}, [], ""):
        errors.append("eligibility_criteria")
    strategies = _read(spec, "treatment_options", "strategies", default=())
    if len(strategies or ()) != 2:
        errors.append("exactly_two_treatment_options")
    elif len({_identifier(item) for item in strategies}) != 2:
        errors.append("unique_treatment_options")
    if not str(
        _read(spec, "time_zero_definition", "time_zero", default="")
    ).strip():
        errors.append("time_zero_definition")
    follow_up = _read(spec, "follow_up_days", "followup_days", "follow_up", default=None)
    if follow_up is None or not str(follow_up).strip():
        errors.append("follow_up")
    elif isinstance(follow_up, (int, float)) and float(follow_up) <= 0:
        errors.append("positive_follow_up_days")
    if not _read(spec, "outcomes", default=()):
        errors.append("outcomes")
    methods = {
        str(item).casefold()
        for item in (_read(spec, "analysis_methods", default=()) or ())
    }
    analysis_plan = str(_read(spec, "analysis_plan", default="")).casefold()
    combined_methods = " ".join(methods) + " " + analysis_plan
    has_weighted = "iptw" in combined_methods or "marginal" in combined_methods
    has_doubly_robust = "aipw" in combined_methods or "doubly" in combined_methods
    if not (has_weighted and has_doubly_robust):
        errors.append("supported_analysis_method")
    if not str(_read(spec, "ood_definition", default="")).strip():
        errors.append("ood_definition")
    if not str(_read(spec, "negative_control_outcome", default="")).strip():
        errors.append("negative_control_outcome")
    if not _read(spec, "sensitivity_analyses", default=()):
        errors.append("sensitivity_analyses")
    if not bool(_read(spec, "locked", "preregistered", default=False)):
        errors.append("locked_or_preregistered_protocol")
    if errors:
        raise CausalAnalysisError(
            "target trial specification is incomplete: " + ", ".join(errors)
        )


@dataclass(frozen=True, slots=True)
class CausalAdmissionThresholds:
    minimum_eligible_patients: int = 1_000
    minimum_raw_per_action: int = 200
    minimum_common_support_coverage: float = 0.80
    minimum_effective_sample_size_per_action: float = 100.0
    propensity_floor: float = 0.05
    maximum_abs_weighted_smd: float = 0.10
    maximum_ci_width: float = 1.00
    maximum_ood_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_eligible_patients < 1_000:
            raise CausalAnalysisError("eligible-patient gate cannot be below 1000")
        if self.minimum_raw_per_action < 200:
            raise CausalAnalysisError("raw action gate cannot be below 200")
        if not 0.80 <= self.minimum_common_support_coverage <= 1:
            raise CausalAnalysisError("common-support gate cannot be below 0.80")
        if self.minimum_effective_sample_size_per_action < 100:
            raise CausalAnalysisError("effective sample-size gate cannot be below 100")
        if not 0.05 <= self.propensity_floor < 0.5:
            raise CausalAnalysisError("propensity floor cannot be below 0.05")
        if not 0 < self.maximum_abs_weighted_smd <= 0.10:
            raise CausalAnalysisError("weighted SMD gate cannot exceed 0.10")
        if not 0 < self.maximum_ci_width <= 1.0:
            raise CausalAnalysisError("confidence-interval width gate cannot exceed 1.0")
        if self.maximum_ood_fraction != 0.0:
            raise CausalAnalysisError("OOD tolerance must remain zero")


@dataclass(frozen=True, slots=True)
class AdmissionReport:
    admitted: bool
    abstained: bool
    reasons: tuple[str, ...]
    eligible_patient_count: int
    raw_action_counts: Mapping[str, int]
    common_support_coverage: float | None = None
    effective_sample_sizes: Mapping[str, float] = field(default_factory=dict)
    maximum_abs_weighted_smd: float | None = None
    ood_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class ResearchEffect:
    method: str
    treatment_id: str
    comparator_id: str
    outcome: str
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    max_abs_weighted_smd: float
    common_support_coverage: float
    effective_sample_sizes: Mapping[str, float]
    core_model: CausalEffectEstimate | None = None
    candidate_only: bool = True
    causal_claim_permitted: bool = False
    disclaimer: str = RESEARCH_ONLY_DISCLAIMER

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower


@dataclass(frozen=True, slots=True)
class ParetoTreatmentOption:
    treatment_id: str
    metrics: Mapping[str, float]
    pareto_optimal: bool
    raw_sample_size: int
    core_model: TreatmentOption | None = None
    is_order: bool = False


@dataclass(frozen=True, slots=True)
class CausalStudyResult:
    target_trial_spec_id: str
    admission: AdmissionReport
    effects: tuple[ResearchEffect, ...]
    pareto_options: tuple[ParetoTreatmentOption, ...]
    applicability: ApplicabilityReport | None
    abstained: bool
    abstention_reasons: tuple[str, ...]
    direction_consistent: bool | None
    candidate_only: bool = True
    causal_claim_permitted: bool = False
    disclaimer: str = RESEARCH_ONLY_DISCLAIMER


def _safe_model(model_type: type[Any], candidates: Mapping[str, Any]) -> Any | None:
    fields = getattr(model_type, "model_fields", {})
    payload = {name: value for name, value in candidates.items() if name in fields}
    try:
        return model_type.model_validate(payload)
    except Exception:
        return None


def _as_float_matrix(
    rows: Sequence[Mapping[str, Any]], covariates: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    dict_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows):
        encoded: dict[str, Any] = {}
        for name in covariates:
            if name not in row or row[name] is None:
                raise CausalAnalysisError(
                    f"missing confounder {name!r} at eligible row {row_number}"
                )
            value = row[name]
            if isinstance(value, (bool, str)):
                encoded[name] = str(value)
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise CausalAnalysisError(
                        f"unsupported confounder value for {name!r}"
                    ) from exc
                if not np.isfinite(numeric):
                    raise CausalAnalysisError(f"confounder {name!r} must be finite")
                encoded[name] = numeric
        dict_rows.append(encoded)
    vectorizer = DictVectorizer(sparse=False)
    matrix = np.asarray(vectorizer.fit_transform(dict_rows), dtype=float)
    names = tuple(str(name) for name in vectorizer.get_feature_names_out())
    return matrix, names


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.sum(np.square(weights)))
    return float(np.sum(weights) ** 2 / denominator) if denominator > 0 else 0.0


def _weighted_mean_and_var(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(weights))
    if total <= 0:
        return math.nan, math.nan
    mean = float(np.sum(weights * values) / total)
    variance = float(np.sum(weights * np.square(values - mean)) / total)
    return mean, variance


def weighted_standardized_mean_differences(
    matrix: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
) -> Mapping[str, float]:
    """Calculate absolute post-weighting SMD for each encoded confounder."""

    names = feature_names or tuple(f"x{index}" for index in range(matrix.shape[1]))
    if len(names) != matrix.shape[1]:
        raise CausalAnalysisError("feature_names does not match matrix width")
    result: dict[str, float] = {}
    for index, name in enumerate(names):
        treated_mean, treated_var = _weighted_mean_and_var(
            matrix[treatment == 1, index], weights[treatment == 1]
        )
        control_mean, control_var = _weighted_mean_and_var(
            matrix[treatment == 0, index], weights[treatment == 0]
        )
        pooled = math.sqrt(max((treated_var + control_var) / 2.0, 0.0))
        if not np.isfinite(pooled) or pooled <= 1e-12:
            difference = 0.0 if math.isclose(treated_mean, control_mean) else math.inf
        else:
            difference = abs(treated_mean - control_mean) / pooled
        result[str(name)] = float(difference)
    return result


def _fit_outcome_model(
    matrix: np.ndarray,
    outcome: np.ndarray,
    *,
    random_seed: int,
) -> Any:
    unique = np.unique(outcome)
    if set(unique.tolist()).issubset({0.0, 1.0}) and len(unique) == 2:
        return LogisticRegression(max_iter=2_000, random_state=random_seed).fit(
            matrix, outcome.astype(int)
        )
    return Ridge(alpha=1.0, random_state=random_seed).fit(matrix, outcome)


def _predict_outcome(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    return np.asarray(model.predict(matrix), dtype=float)


def _constant_or_model_prediction(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    random_seed: int,
) -> np.ndarray:
    if len(train_y) == 0:
        raise CausalAnalysisError("an outcome model has no training observations")
    if np.all(train_y == train_y[0]):
        return np.full(len(test_x), float(train_y[0]), dtype=float)
    model = _fit_outcome_model(train_x, train_y, random_seed=random_seed)
    return _predict_outcome(model, test_x)


def _cross_fitted_nuisance(
    matrix: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    folds: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum_group = int(min(np.sum(treatment == 0), np.sum(treatment == 1)))
    n_splits = min(folds, minimum_group)
    if n_splits < 2:
        raise CausalAnalysisError("cross-fitting requires at least two patients per action")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_seed
    )
    propensity = np.empty(len(outcome), dtype=float)
    mu_treated = np.empty(len(outcome), dtype=float)
    mu_control = np.empty(len(outcome), dtype=float)
    for fold_index, (train, test) in enumerate(splitter.split(matrix, treatment)):
        scaler = StandardScaler().fit(matrix[train])
        train_x = scaler.transform(matrix[train])
        test_x = scaler.transform(matrix[test])
        propensity_model = LogisticRegression(
            max_iter=2_000,
            random_state=random_seed + fold_index,
        ).fit(train_x, treatment[train])
        propensity[test] = propensity_model.predict_proba(test_x)[:, 1]
        treated_train = treatment[train] == 1
        control_train = ~treated_train
        mu_treated[test] = _constant_or_model_prediction(
            train_x[treated_train],
            outcome[train][treated_train],
            test_x,
            random_seed=random_seed + fold_index,
        )
        mu_control[test] = _constant_or_model_prediction(
            train_x[control_train],
            outcome[train][control_train],
            test_x,
            random_seed=random_seed + fold_index + 101,
        )
    return propensity, mu_treated, mu_control


def _effect_from_scores(scores: np.ndarray) -> tuple[float, float, float, float]:
    estimate = float(np.mean(scores))
    standard_error = float(np.std(scores, ddof=1) / math.sqrt(len(scores)))
    ci_lower = estimate - 1.96 * standard_error
    ci_upper = estimate + 1.96 * standard_error
    return estimate, standard_error, ci_lower, ci_upper


def pareto_treatment_options(
    rows: Sequence[Mapping[str, Any]],
    *,
    action_column: str,
    metric_directions: Mapping[str, Literal["maximize", "minimize"]],
    treatment_options: Sequence[Any] = (),
    decision_point: Any | None = None,
) -> tuple[ParetoTreatmentOption, ...]:
    """Return all non-dominated options; deliberately never choose one winner."""

    if not metric_directions:
        return ()
    actions = sorted({str(row[action_column]) for row in rows})
    metrics: dict[str, dict[str, float]] = {}
    for action in actions:
        group = [row for row in rows if str(row[action_column]) == action]
        values: dict[str, float] = {}
        for metric in metric_directions:
            observed = [float(row[metric]) for row in group if row.get(metric) is not None]
            if observed:
                values[metric] = float(np.mean(observed))
        metrics[action] = values

    option_by_id = {_identifier(option): option for option in treatment_options}
    output: list[ParetoTreatmentOption] = []
    for action in actions:
        dominated = False
        if set(metrics[action]) == set(metric_directions):
            for other in actions:
                if other == action or set(metrics[other]) != set(metric_directions):
                    continue
                weakly_better = True
                strictly_better = False
                for metric, direction in metric_directions.items():
                    left = metrics[other][metric]
                    right = metrics[action][metric]
                    if direction == "maximize":
                        weakly_better &= left >= right
                        strictly_better |= left > right
                    else:
                        weakly_better &= left <= right
                        strictly_better |= left < right
                if weakly_better and strictly_better:
                    dominated = True
                    break
        else:
            dominated = True
        original = option_by_id.get(action)
        core = _safe_model(
            TreatmentOption,
            {
                "id": action,
                "regimen_class": _read(original, "regimen_class", default=action),
                "decision_point": decision_point,
                "effect_estimate_ids": [],
                "benefits": {
                    name: value
                    for name, value in metrics[action].items()
                    if metric_directions[name] == "maximize"
                },
                "harms": {
                    name: value
                    for name, value in metrics[action].items()
                    if metric_directions[name] == "minimize"
                },
                "pareto_optimal": not dominated,
                "rank": None,
                "guideline_backed": True,
                "abstained": dominated,
                "abstention_reasons": ["PARETO_DOMINATED"] if dominated else [],
                "automatic_order_allowed": False,
            },
        )
        output.append(
            ParetoTreatmentOption(
                treatment_id=action,
                metrics=metrics[action],
                pareto_optimal=not dominated,
                raw_sample_size=sum(str(row[action_column]) == action for row in rows),
                core_model=core,
            )
        )
    return tuple(output)


class CausalStrategyEngine:
    """Binary-action IPTW MSM and cross-fitted doubly robust estimator."""

    def __init__(
        self,
        *,
        thresholds: CausalAdmissionThresholds | None = None,
        cross_fit_folds: int = 5,
        random_seed: int = 29,
    ) -> None:
        if cross_fit_folds < 2:
            raise CausalAnalysisError("cross_fit_folds must be at least two")
        self.thresholds = thresholds or CausalAdmissionThresholds()
        self.cross_fit_folds = int(cross_fit_folds)
        self.random_seed = int(random_seed)

    def run(
        self,
        spec: TargetTrialSpec,
        rows: Sequence[Mapping[str, Any]],
        *,
        action_column: str = "treatment_id",
        outcome_column: str | None = None,
        patient_id_column: str = "patient_id",
        ood_column: str | None = "is_ood",
        confounders: Sequence[str] | None = None,
        metric_directions: Mapping[str, Literal["maximize", "minimize"]] | None = None,
        negative_control_passed: bool = False,
        sensitivity_analysis_passed: bool = False,
        decision_time_column: str = "decision_time",
        confounder_availability_suffix: str = "_available_at",
    ) -> CausalStudyResult:
        validate_target_trial_spec(spec)
        spec_id = str(_read(spec, "id", default="target_trial"))
        strategies = tuple(_read(spec, "treatment_options", "strategies", default=()))
        strategy_ids = tuple(_identifier(item) for item in strategies)
        outcome = outcome_column or str(tuple(_read(spec, "outcomes"))[0])
        if outcome not in spec.outcomes:
            raise CausalAnalysisError("outcome must be pre-specified in the target trial")
        output_metrics = set((metric_directions or {}).keys()) | {outcome}
        if not output_metrics.issubset(spec.outcomes):
            raise CausalAnalysisError(
                "every reported benefit/harm metric must be a pre-specified outcome"
            )
        selected_confounders = tuple(
            str(item) for item in (confounders or _read(spec, "confounders", default=()))
        )

        if not rows:
            admission = AdmissionReport(
                admitted=False,
                abstained=True,
                reasons=("NO_ELIGIBLE_PATIENTS",),
                eligible_patient_count=0,
                raw_action_counts={},
            )
            return self._abstention(spec_id, admission)

        if not decision_time_column.strip() or not confounder_availability_suffix:
            raise CausalAnalysisError(
                "decision-time and confounder-availability provenance fields are required"
            )

        if not selected_confounders:
            excluded = {
                patient_id_column,
                action_column,
                outcome,
                decision_time_column,
                *(metric_directions or {}).keys(),
            }
            if ood_column:
                excluded.add(ood_column)
            selected_confounders = tuple(
                sorted(
                    name
                    for name in rows[0]
                    if name not in excluded
                    and not name.endswith(confounder_availability_suffix)
                    and all(row.get(name) is not None for row in rows)
                )
            )
        if not selected_confounders:
            raise CausalAnalysisError(
                "at least one pre-treatment confounder is required for adjustment"
            )

        provenance_reasons: set[str] = set()
        for row in rows:
            try:
                decision_time = _as_utc_timepoint(row.get(decision_time_column))
            except CausalAnalysisError:
                provenance_reasons.add("TIME_ZERO_PROVENANCE_MISSING_OR_INVALID")
                continue
            for confounder in selected_confounders:
                availability_column = f"{confounder}{confounder_availability_suffix}"
                try:
                    available_at = _as_utc_timepoint(row.get(availability_column))
                except CausalAnalysisError:
                    provenance_reasons.add("CONFOUNDER_AVAILABILITY_MISSING_OR_INVALID")
                    continue
                if available_at > decision_time:
                    provenance_reasons.add("POST_TIME_ZERO_CONFOUNDER")
            action_assigned_at: datetime | None = None
            try:
                action_assigned_at = _as_utc_timepoint(
                    row.get(spec.action_assigned_at_column)
                )
            except CausalAnalysisError:
                provenance_reasons.add("ACTION_ASSIGNMENT_TIME_MISSING_OR_INVALID")
            else:
                grace_seconds = spec.assignment_grace_period_days * 86_400
                elapsed = (action_assigned_at - decision_time).total_seconds()
                if elapsed < 0 or elapsed > grace_seconds:
                    provenance_reasons.add(
                        "ACTION_ASSIGNMENT_OUTSIDE_TIME_ZERO_WINDOW"
                    )
            for metric in output_metrics:
                outcome_time_column = spec.outcome_observed_at_columns[metric]
                try:
                    outcome_observed_at = _as_utc_timepoint(
                        row.get(outcome_time_column)
                    )
                except CausalAnalysisError:
                    provenance_reasons.add(
                        "OUTCOME_OBSERVATION_TIME_MISSING_OR_INVALID"
                    )
                else:
                    elapsed = (outcome_observed_at - decision_time).total_seconds()
                    if (
                        elapsed <= 0
                        or elapsed > spec.follow_up_days * 86_400
                        or (
                            action_assigned_at is not None
                            and outcome_observed_at <= action_assigned_at
                        )
                    ):
                        provenance_reasons.add(
                            "OUTCOME_OBSERVATION_OUTSIDE_FOLLOW_UP_WINDOW"
                        )

        if provenance_reasons:
            action_counts = Counter(
                str(row.get(action_column, "")).strip()
                for row in rows
                if str(row.get(action_column, "")).strip()
            )
            admission = AdmissionReport(
                admitted=False,
                abstained=True,
                reasons=tuple(sorted(provenance_reasons)),
                eligible_patient_count=len(rows),
                raw_action_counts=dict(action_counts),
            )
            return self._abstention(spec_id, admission)

        ids = [str(row.get(patient_id_column, "")).strip() for row in rows]
        if any(not item for item in ids):
            raise CausalAnalysisError("each eligible row requires a patient id")
        if len(ids) != len(set(ids)):
            raise CausalAnalysisError(
                "target-trial analytic input must contain one time-zero row per patient"
            )

        action_values = [str(row.get(action_column, "")).strip() for row in rows]
        unsupported = sorted(set(action_values) - set(strategy_ids))
        if unsupported or any(not item for item in action_values):
            raise CausalAnalysisError(
                "observed actions must match the two locked protocol options"
            )
        counts = Counter(action_values)
        reasons: list[str] = []
        if len(rows) < self.thresholds.minimum_eligible_patients:
            reasons.append("INSUFFICIENT_ELIGIBLE_PATIENTS")
        if any(
            counts[action] < self.thresholds.minimum_raw_per_action
            for action in strategy_ids
        ):
            reasons.append("INSUFFICIENT_RAW_ACTION_COUNT")
        ood_status_missing = not ood_column or any(
            ood_column not in row or row.get(ood_column) is None for row in rows
        )
        if ood_status_missing:
            reasons.append("OOD_STATUS_MISSING")
            ood_fraction = 1.0
        elif any(
            not isinstance(row[ood_column], (bool, np.bool_))  # type: ignore[index]
            for row in rows
        ):
            reasons.append("OOD_STATUS_INVALID")
            ood_fraction = 1.0
        else:
            assert ood_column is not None
            ood_fraction = float(np.mean([bool(row[ood_column]) for row in rows]))
        if ood_fraction > self.thresholds.maximum_ood_fraction:
            reasons.append("OUT_OF_DISTRIBUTION_INPUT")
        early_admission = AdmissionReport(
            admitted=not reasons,
            abstained=bool(reasons),
            reasons=tuple(reasons),
            eligible_patient_count=len(rows),
            raw_action_counts=dict(counts),
            ood_fraction=ood_fraction,
        )
        if reasons:
            return self._abstention(spec_id, early_admission)

        matrix, encoded_names = _as_float_matrix(rows, selected_confounders)
        treatment = np.asarray(
            [1 if action == strategy_ids[0] else 0 for action in action_values],
            dtype=int,
        )
        outcomes = np.asarray([float(row[outcome]) for row in rows], dtype=float)
        if not np.all(np.isfinite(outcomes)):
            raise CausalAnalysisError("outcomes must be present and finite")
        propensity, mu_treated, mu_control = _cross_fitted_nuisance(
            matrix,
            treatment,
            outcomes,
            folds=self.cross_fit_folds,
            random_seed=self.random_seed,
        )
        floor = self.thresholds.propensity_floor
        support_mask = (propensity >= floor) & (propensity <= 1 - floor)
        coverage = float(np.mean(support_mask))
        clipped = np.clip(propensity, floor, 1 - floor)
        marginal = float(np.mean(treatment))
        weights = np.where(
            treatment == 1,
            marginal / clipped,
            (1 - marginal) / (1 - clipped),
        )
        ess = {
            strategy_ids[0]: _effective_sample_size(weights[treatment == 1]),
            strategy_ids[1]: _effective_sample_size(weights[treatment == 0]),
        }
        smds = weighted_standardized_mean_differences(
            matrix, treatment, weights, feature_names=encoded_names
        )
        max_smd = max(smds.values(), default=0.0)
        gate_reasons: list[str] = []
        if coverage < self.thresholds.minimum_common_support_coverage:
            gate_reasons.append("POSITIVITY_OR_COMMON_SUPPORT_FAILURE")
        if any(
            value < self.thresholds.minimum_effective_sample_size_per_action
            for value in ess.values()
        ):
            gate_reasons.append("INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE")
        if not np.isfinite(max_smd) or max_smd >= self.thresholds.maximum_abs_weighted_smd:
            gate_reasons.append("POST_WEIGHTING_IMBALANCE")

        admission = AdmissionReport(
            admitted=not gate_reasons,
            abstained=bool(gate_reasons),
            reasons=tuple(gate_reasons),
            eligible_patient_count=len(rows),
            raw_action_counts=dict(counts),
            common_support_coverage=coverage,
            effective_sample_sizes=ess,
            maximum_abs_weighted_smd=float(max_smd),
            ood_fraction=ood_fraction,
        )
        if gate_reasons:
            return self._abstention(spec_id, admission)

        aipw_scores = (
            mu_treated
            - mu_control
            + treatment * (outcomes - mu_treated) / clipped
            - (1 - treatment) * (outcomes - mu_control) / (1 - clipped)
        )
        aipw_values = _effect_from_scores(aipw_scores)

        treated_normaliser = float(np.mean(treatment / clipped))
        control_normaliser = float(np.mean((1 - treatment) / (1 - clipped)))
        treated_mean = float(
            np.mean(treatment * outcomes / clipped) / treated_normaliser
        )
        control_mean = float(
            np.mean((1 - treatment) * outcomes / (1 - clipped)) / control_normaliser
        )
        iptw_scores = (
            treatment / clipped * (outcomes - treated_mean) / treated_normaliser
            - (1 - treatment)
            / (1 - clipped)
            * (outcomes - control_mean)
            / control_normaliser
            + treated_mean
            - control_mean
        )
        iptw_values = _effect_from_scores(iptw_scores)
        direction_consistent = bool(np.sign(aipw_values[0]) == np.sign(iptw_values[0]))

        effects: list[ResearchEffect] = []
        for method, values in (
            ("cross_fitted_aipw", aipw_values),
            ("iptw_marginal_structural_model", iptw_values),
        ):
            estimate, standard_error, lower, upper = values
            applicability = self._applicability(
                status="applicable",
                reasons=(),
                coverage=coverage,
                ci_width=upper - lower,
                can_present=True,
            )
            core = _safe_model(
                CausalEffectEstimate,
                {
                    "target_trial_spec_id": spec_id,
                    "target_trial_id": spec_id,
                    "strategy": strategy_ids[0],
                    "comparator": strategy_ids[1],
                    "outcome": outcome,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "method": method,
                    "corroborating_methods": [
                        "iptw_marginal_structural_model"
                        if method == "cross_fitted_aipw"
                        else "cross_fitted_aipw"
                    ],
                    "eligible_patients": len(rows),
                    "strategy_patients": counts[strategy_ids[0]],
                    "comparator_patients": counts[strategy_ids[1]],
                    "common_support_fraction": coverage,
                    "strategy_effective_sample_size": ess[strategy_ids[0]],
                    "comparator_effective_sample_size": ess[strategy_ids[1]],
                    "maximum_absolute_smd": float(max_smd),
                    "sensitivity_analysis": {
                        "passed": str(sensitivity_analysis_passed).lower()
                    },
                    "negative_controls_passed": negative_control_passed,
                    "directionally_consistent": direction_consistent,
                    "abstained": False,
                    "abstention_reasons": [],
                    "research_only": True,
                    "clinical_efficacy_claim": False,
                },
            )
            effects.append(
                ResearchEffect(
                    method=method,
                    treatment_id=strategy_ids[0],
                    comparator_id=strategy_ids[1],
                    outcome=outcome,
                    estimate=estimate,
                    standard_error=standard_error,
                    ci_lower=lower,
                    ci_upper=upper,
                    max_abs_weighted_smd=float(max_smd),
                    common_support_coverage=coverage,
                    effective_sample_sizes=ess,
                    core_model=core,
                )
            )

        final_reasons: list[str] = []
        if any(effect.ci_width > self.thresholds.maximum_ci_width for effect in effects):
            final_reasons.append("CONFIDENCE_INTERVAL_TOO_WIDE")
        if not direction_consistent:
            final_reasons.append("ESTIMATOR_DIRECTION_INCONSISTENT")
        if not negative_control_passed:
            final_reasons.append("NEGATIVE_CONTROL_FAILED")
        if not sensitivity_analysis_passed:
            final_reasons.append("SENSITIVITY_ANALYSIS_FAILED")
        pareto = pareto_treatment_options(
            rows,
            action_column=action_column,
            metric_directions=metric_directions or {},
            treatment_options=strategies,
            decision_point=_read(spec, "decision_point", "decision_point_kind"),
        )
        can_present = not final_reasons
        applicability = self._applicability(
            status="applicable" if can_present else "abstain",
            reasons=tuple(final_reasons),
            coverage=coverage,
            ci_width=max(effect.ci_width for effect in effects),
            can_present=can_present,
        )
        return CausalStudyResult(
            target_trial_spec_id=spec_id,
            admission=admission,
            effects=() if final_reasons else tuple(effects),
            pareto_options=pareto if can_present else (),
            applicability=applicability,
            abstained=bool(final_reasons),
            abstention_reasons=tuple(final_reasons),
            direction_consistent=direction_consistent,
        )

    def _applicability(
        self,
        *,
        status: str,
        reasons: Sequence[str],
        coverage: float | None,
        ci_width: float | None,
        can_present: bool,
    ) -> ApplicabilityReport | None:
        return _safe_model(
            ApplicabilityReport,
            {
                "recommendation_id": "target_trial_applicability",
                "applicable": can_present,
                "matched_criteria": (
                    ["common_support", "effective_sample_size", "covariate_balance"]
                    if can_present
                    else []
                ),
                "failed_criteria": list(reasons),
                "missing_critical_features": [],
                "ood": "OUT_OF_DISTRIBUTION_INPUT" in reasons,
                "abstained": not can_present,
                "abstention_reasons": list(reasons),
                "independent_review_basis": [
                    "locked target-trial specification",
                    "95% confidence interval",
                    f"common support={coverage}" if coverage is not None else "common support unavailable",
                    f"CI width={ci_width}" if ci_width is not None else "CI unavailable",
                ],
                "automatic_order_allowed": False,
            },
        )

    def _abstention(
        self, spec_id: str, admission: AdmissionReport
    ) -> CausalStudyResult:
        applicability = self._applicability(
            status="abstain",
            reasons=admission.reasons,
            coverage=admission.common_support_coverage,
            ci_width=None,
            can_present=False,
        )
        return CausalStudyResult(
            target_trial_spec_id=spec_id,
            admission=admission,
            effects=(),
            pareto_options=(),
            applicability=applicability,
            abstained=True,
            abstention_reasons=admission.reasons,
            direction_consistent=None,
        )


TargetTrialEngine = CausalStrategyEngine


def lock_target_trial_result(
    spec: TargetTrialSpec,
    result: CausalStudyResult,
    *,
    negative_control_audit: CausalAuditArtifact,
    sensitivity_audits: Sequence[CausalAuditArtifact],
    input_snapshot: TargetTrialInputSnapshot,
    result_id: str | None = None,
) -> TargetTrialResult:
    """Create a hash-bindable result only after every causal audit has passed."""

    if result.abstained or not result.admission.admitted or len(result.effects) < 2:
        raise CausalAnalysisError("an abstained or incomplete analysis cannot be locked")
    identifier = result_id or new_id("trial_result")
    spec_hash = spec.content_hash()
    effects: list[CausalEffectEstimate] = []
    for research_effect in result.effects:
        if research_effect.core_model is None:
            raise CausalAnalysisError("locked result requires validated core effects")
        payload = research_effect.core_model.model_dump(mode="json")
        payload.update(
            {
                "target_trial_result_id": identifier,
                "target_trial_spec_hash": spec_hash,
            }
        )
        effects.append(CausalEffectEstimate.model_validate(payload))
    if result.admission.common_support_coverage is None:
        raise CausalAnalysisError("locked result requires common support evidence")
    if result.admission.maximum_abs_weighted_smd is None:
        raise CausalAnalysisError("locked result requires covariate balance evidence")
    return TargetTrialResult(
        id=identifier,
        spec=spec,
        effects=effects,
        eligible_patient_count=result.admission.eligible_patient_count,
        common_support_fraction=result.admission.common_support_coverage,
        maximum_absolute_smd=result.admission.maximum_abs_weighted_smd,
        ood_fraction=result.admission.ood_fraction,
        input_snapshot=input_snapshot,
        negative_control_audit=negative_control_audit,
        sensitivity_audits=list(sensitivity_audits),
        negative_control_passed=True,
        sensitivity_analysis_passed=True,
        directionally_consistent=True,
        abstained=False,
        locked_at=datetime.now(timezone.utc),
    )


__all__ = [
    "AdmissionReport",
    "CausalAdmissionThresholds",
    "CausalAnalysisError",
    "CausalStrategyEngine",
    "CausalStudyResult",
    "ParetoTreatmentOption",
    "RESEARCH_ONLY_DISCLAIMER",
    "ResearchEffect",
    "TargetTrialEngine",
    "pareto_treatment_options",
    "lock_target_trial_result",
    "validate_target_trial_spec",
    "weighted_standardized_mean_differences",
]
