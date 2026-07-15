"""Deterministic, research-only longitudinal disease-state discovery.

The implementation deliberately uses an interpretable Gaussian mixture rather
than an end-to-end treatment recommender.  Treatment is an *external action*
and is never accepted as a state feature.  Transition intensities are estimated
per unit of observed person-time so irregularly sampled clinical timelines can
be compared without pretending they are evenly spaced.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score, brier_score_loss
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ai_clinician.feature_manifest import (
    AnalysisColumn,
    AnalysisColumnManifest,
    ColumnRole,
    ManifestValidationError,
    canonical_manifest_sha256,
    parse_analysis_manifest,
)
from ai_clinician.models import CandidateDiseaseState, StateTransition


_TREATMENT_TOKENS = frozenset(
    {
        "treatment",
        "therapy",
        "regimen",
        "drug",
        "medication",
        "dose",
        "chemo",
        "radiation",
        "surgery",
        "line_of_therapy",
        "lot",
        "治疗",
        "用药",
        "药物",
        "方案",
        "剂量",
        "化疗",
        "放疗",
        "手术",
        "线次",
        "folfox",
        "folfiri",
        "capox",
        "xelox",
        "bevacizumab",
        "cetuximab",
        "panitumumab",
        "regorafenib",
        "fruquintinib",
        "贝伐珠单抗",
        "西妥昔单抗",
        "帕尼单抗",
        "瑞戈非尼",
        "呋喹替尼",
    }
)


class StateDiscoveryError(ValueError):
    """Raised when state discovery would be invalid or non-reproducible."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, np.datetime64):
        value = str(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError as exc:
            raise StateDiscoveryError(f"invalid ISO date/time: {value!r}") from exc
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    raise StateDiscoveryError(f"unsupported date/time value: {value!r}")


def _normalise_feature_name(name: str) -> str:
    return "_".join(name.casefold().strip().replace("-", "_").split())


def _is_treatment_feature(name: str) -> bool:
    normalised = _normalise_feature_name(name)
    return any(token in normalised for token in _TREATMENT_TOKENS)


@dataclass(frozen=True, slots=True)
class PatientPartition:
    """Patient-level temporal partition with no cross-split patient leakage."""

    development: tuple[str, ...]
    tuning: tuple[str, ...]
    locked_test: tuple[str, ...]
    index_dates: Mapping[str, datetime] = field(repr=False)

    def __post_init__(self) -> None:
        groups = [set(self.development), set(self.tuning), set(self.locked_test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise StateDiscoveryError("a patient cannot appear in more than one partition")
        if set().union(*groups) != set(self.index_dates):
            raise StateDiscoveryError("partition membership must cover each patient exactly once")

    @property
    def all_patient_ids(self) -> tuple[str, ...]:
        return self.development + self.tuning + self.locked_test

    def split_for(self, patient_id: str) -> str:
        if patient_id in self.development:
            return "development"
        if patient_id in self.tuning:
            return "tuning"
        if patient_id in self.locked_test:
            return "locked_test"
        raise KeyError(patient_id)


def chronological_patient_split(
    records: Sequence[Mapping[str, Any]],
    *,
    patient_id_column: str = "patient_id",
    index_date_column: str = "decision_date",
    development_fraction: float = 0.70,
    tuning_fraction: float = 0.15,
) -> PatientPartition:
    """Split whole patients by their earliest mCRC decision date.

    Stable patient-id ordering breaks date ties.  Fractions are applied to the
    patient count, never the row count, so every event for one patient remains
    in exactly one partition.
    """

    if not records:
        raise StateDiscoveryError("at least one record is required")
    if not (0 < development_fraction < 1):
        raise StateDiscoveryError("development_fraction must be between 0 and 1")
    if not (0 <= tuning_fraction < 1):
        raise StateDiscoveryError("tuning_fraction must be between 0 and 1")
    if development_fraction + tuning_fraction >= 1:
        raise StateDiscoveryError("development and tuning fractions must sum to less than 1")

    index_dates: dict[str, datetime] = {}
    for row_number, record in enumerate(records):
        raw_patient_id = record.get(patient_id_column)
        if raw_patient_id is None or not str(raw_patient_id).strip():
            raise StateDiscoveryError(f"record {row_number} has no patient id")
        if index_date_column not in record or record[index_date_column] is None:
            raise StateDiscoveryError(f"record {row_number} has no index date")
        patient_id = str(raw_patient_id).strip()
        decision_date = _as_datetime(record[index_date_column])
        if patient_id not in index_dates or decision_date < index_dates[patient_id]:
            index_dates[patient_id] = decision_date

    ordered = sorted(index_dates, key=lambda pid: (index_dates[pid], pid))
    n_patients = len(ordered)
    development_end = math.floor(n_patients * development_fraction)
    tuning_end = development_end + math.floor(n_patients * tuning_fraction)
    # Preserve all three sets for small synthetic feasibility checks whenever
    # possible, without changing the 70/15/15 rule for normal cohorts.
    if n_patients >= 3:
        development_end = max(1, min(development_end, n_patients - 2))
        tuning_end = max(development_end + 1, min(tuning_end, n_patients - 1))
    return PatientPartition(
        development=tuple(ordered[:development_end]),
        tuning=tuple(ordered[development_end:tuning_end]),
        locked_test=tuple(ordered[tuning_end:]),
        index_dates=index_dates,
    )


def assert_no_future_information(
    records: Sequence[Mapping[str, Any]],
    *,
    event_time_column: str = "event_time",
    feature_available_time_column: str = "feature_available_time",
) -> None:
    """Reject decision features that became available after the decision/event."""

    for row_number, record in enumerate(records):
        if event_time_column not in record or record.get(event_time_column) is None:
            raise StateDiscoveryError(
                f"record {row_number} has no decision/event time for leakage audit"
            )
        if (
            feature_available_time_column not in record
            or record.get(feature_available_time_column) is None
        ):
            raise StateDiscoveryError(
                f"record {row_number} has no feature availability time"
            )
        available = record.get(feature_available_time_column)
        decision = record.get(event_time_column)
        if _as_datetime(available) > _as_datetime(decision):
            raise StateDiscoveryError(
                f"future-information leakage at record {row_number}: "
                f"{feature_available_time_column} is after {event_time_column}"
            )


@dataclass(frozen=True, slots=True)
class StateValidationThresholds:
    minimum_bootstrap_ari: float = 0.75
    minimum_calibration_slope: float = 0.80
    maximum_calibration_slope: float = 1.20
    minimum_relative_brier_improvement: float = 0.05
    minimum_expert_confirmations: int = 2


@dataclass(frozen=True, slots=True)
class StateValidationGate:
    bootstrap_ari: float
    calibration_slope: float
    model_brier: float
    baseline_brier: float
    relative_brier_improvement: float
    expert_confirmations: int
    passed: bool
    reasons: tuple[str, ...]
    thresholds: StateValidationThresholds


def evaluate_state_validation_gate(
    *,
    bootstrap_ari: float,
    calibration_slope: float,
    model_brier: float,
    baseline_brier: float,
    expert_confirmations: int,
    thresholds: StateValidationThresholds | None = None,
) -> StateValidationGate:
    """Apply all pre-specified state-promotion gates without discretion."""

    limits = thresholds or StateValidationThresholds()
    if baseline_brier <= 0:
        relative_improvement = 0.0 if model_brier == baseline_brier else -math.inf
    else:
        relative_improvement = (baseline_brier - model_brier) / baseline_brier
    reasons: list[str] = []
    if not np.isfinite(bootstrap_ari) or bootstrap_ari < limits.minimum_bootstrap_ari:
        reasons.append("STATE_STABILITY_BELOW_THRESHOLD")
    if not np.isfinite(calibration_slope) or not (
        limits.minimum_calibration_slope
        <= calibration_slope
        <= limits.maximum_calibration_slope
    ):
        reasons.append("CALIBRATION_SLOPE_OUT_OF_RANGE")
    if (
        not np.isfinite(relative_improvement)
        or relative_improvement < limits.minimum_relative_brier_improvement
    ):
        reasons.append("BRIER_IMPROVEMENT_BELOW_THRESHOLD")
    if expert_confirmations < limits.minimum_expert_confirmations:
        reasons.append("INSUFFICIENT_EXPERT_INTERPRETABILITY_CONFIRMATION")
    return StateValidationGate(
        bootstrap_ari=float(bootstrap_ari),
        calibration_slope=float(calibration_slope),
        model_brier=float(model_brier),
        baseline_brier=float(baseline_brier),
        relative_brier_improvement=float(relative_improvement),
        expert_confirmations=int(expert_confirmations),
        passed=not reasons,
        reasons=tuple(reasons),
        thresholds=limits,
    )


@runtime_checkable
class StateModelChallenger(Protocol):
    """Minimal interface for a time-aware challenger implementation."""

    name: str

    def fit(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        feature_names: Sequence[str],
    ) -> "StateModelChallenger": ...

    def predict_risk(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ChallengerComparison:
    challenger_name: str
    baseline_brier: float
    challenger_brier: float
    relative_improvement: float
    challenger_wins: bool


def compare_challenger(
    challenger: StateModelChallenger,
    records: Sequence[Mapping[str, Any]],
    outcomes: Sequence[int | bool | float],
    baseline_risk: Sequence[float],
    *,
    feature_names: Sequence[str],
) -> ChallengerComparison:
    challenger.fit(records, feature_names=feature_names)
    predicted = np.asarray(challenger.predict_risk(records), dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    baseline = np.asarray(baseline_risk, dtype=float)
    if predicted.shape != observed.shape or baseline.shape != observed.shape:
        raise StateDiscoveryError("challenger, baseline, and outcome vectors must align")
    challenger_brier = float(brier_score_loss(observed, np.clip(predicted, 0, 1)))
    baseline_brier = float(brier_score_loss(observed, np.clip(baseline, 0, 1)))
    improvement = (
        (baseline_brier - challenger_brier) / baseline_brier
        if baseline_brier > 0
        else 0.0
    )
    return ChallengerComparison(
        challenger_name=str(challenger.name),
        baseline_brier=baseline_brier,
        challenger_brier=challenger_brier,
        relative_improvement=float(improvement),
        challenger_wins=challenger_brier < baseline_brier,
    )


@dataclass(slots=True)
class FittedStateModel:
    """Serializable-in-principle fitted state model and interpretable outputs."""

    feature_names: tuple[str, ...]
    treatment_columns: tuple[str, ...]
    analysis_manifest_sha256: str
    analysis_column_roles: Mapping[str, str]
    scaler: StandardScaler
    mixture: GaussianMixture
    state_labels: tuple[str, ...]
    component_order: tuple[int, ...]
    centroids: Mapping[str, Mapping[str, float]]
    transition_rates: np.ndarray
    transition_counts: np.ndarray
    person_time: np.ndarray
    random_seed: int
    candidate_states: tuple[CandidateDiseaseState, ...] = ()
    state_transitions: tuple[StateTransition, ...] = ()
    fitted_at: datetime = field(default_factory=_utcnow)

    def _matrix(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return _feature_matrix(records, self.feature_names)

    def predict(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = self.scaler.transform(self._matrix(records))
        raw = self.mixture.predict(matrix)
        remap = {old: new for new, old in enumerate(self.component_order)}
        return np.asarray([remap[int(label)] for label in raw], dtype=int)

    def predict_proba(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = self.scaler.transform(self._matrix(records))
        return self.mixture.predict_proba(matrix)[:, self.component_order]

    def continuous_state(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return posterior state memberships as a continuous latent state."""

        return self.predict_proba(records)


@dataclass(frozen=True, slots=True)
class StateFitReport:
    model: FittedStateModel
    labels: tuple[int, ...]
    posterior_probabilities: tuple[tuple[float, ...], ...]
    bootstrap_ari: float
    analysis_manifest_sha256: str
    treatment_features_excluded: bool = True
    deterministic: bool = True


def _feature_matrix(
    records: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    rows: list[list[float]] = []
    for row_number, record in enumerate(records):
        row: list[float] = []
        for feature in feature_names:
            value = record.get(feature)
            if value is None:
                raise StateDiscoveryError(
                    f"missing state feature {feature!r} at record {row_number}"
                )
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise StateDiscoveryError(
                    f"state feature {feature!r} must be numeric"
                ) from exc
            if not np.isfinite(numeric):
                raise StateDiscoveryError(
                    f"state feature {feature!r} must be finite"
                )
            row.append(numeric)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _patient_cluster_ids(
    records: Sequence[Mapping[str, Any]], patient_id_column: str
) -> np.ndarray:
    patient_ids: list[str] = []
    for row_number, record in enumerate(records):
        raw_patient_id = record.get(patient_id_column)
        if raw_patient_id is None or not str(raw_patient_id).strip():
            raise StateDiscoveryError(
                f"record {row_number} has no {patient_id_column!r} for cluster bootstrap"
            )
        patient_ids.append(str(raw_patient_id).strip())
    return np.asarray(patient_ids, dtype=object)


def _patient_cluster_bootstrap_indices(
    patient_ids: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    """Resample patients with replacement and retain every row in each cluster."""

    rows_by_patient: defaultdict[str, list[int]] = defaultdict(list)
    for row_index, patient_id in enumerate(patient_ids):
        rows_by_patient[str(patient_id)].append(row_index)
    patients = tuple(sorted(rows_by_patient))
    if not patients:
        raise StateDiscoveryError("patient cluster bootstrap requires at least one patient")
    sampled_positions = rng.choice(len(patients), size=len(patients), replace=True)
    sampled_rows = [
        row_index
        for position in sampled_positions
        for row_index in rows_by_patient[patients[int(position)]]
    ]
    return np.asarray(sampled_rows, dtype=int)


def _safe_model(
    model_type: type[Any], candidates: Mapping[str, Any]
) -> Any | None:
    """Construct a shared Schema while tolerating harmless field-name evolution.

    The core models are authoritative.  This adapter filters engine metadata to
    fields exposed by that Schema; it never bypasses Pydantic validation.
    """

    fields = getattr(model_type, "model_fields", {})
    payload = {name: value for name, value in candidates.items() if name in fields}
    try:
        return model_type.model_validate(payload)
    except Exception:
        return None


class ContinuousTimeStateDiscovery:
    """Interpretable continuous-time mixture-state discovery engine."""

    def __init__(
        self,
        *,
        n_states: int = 3,
        random_seed: int = 17,
        bootstrap_replicates: int = 20,
        minimum_elapsed_days: float = 1.0 / 24.0,
    ) -> None:
        if n_states < 2:
            raise StateDiscoveryError("n_states must be at least two")
        if bootstrap_replicates < 2:
            raise StateDiscoveryError("bootstrap_replicates must be at least two")
        if minimum_elapsed_days <= 0:
            raise StateDiscoveryError("minimum_elapsed_days must be positive")
        self.n_states = int(n_states)
        self.random_seed = int(random_seed)
        self.bootstrap_replicates = int(bootstrap_replicates)
        self.minimum_elapsed_days = float(minimum_elapsed_days)

    def fit(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        feature_names: Sequence[str],
        patient_id_column: str = "patient_id",
        time_column: str = "event_time",
        index_date_column: str = "decision_date",
        feature_available_time_column: str = "feature_available_time",
        treatment_columns: Sequence[str] = (),
        column_manifest: AnalysisColumnManifest | Mapping[str, Any] | None = None,
        manifest_sha256: str | None = None,
    ) -> StateFitReport:
        if len(records) < max(self.n_states * 3, 10):
            raise StateDiscoveryError("too few observations for the requested states")
        if not feature_names:
            raise StateDiscoveryError("at least one state feature is required")
        if len(set(feature_names)) != len(feature_names):
            raise StateDiscoveryError("state feature names must be unique")

        # The content-addressed manifest, not column-name heuristics, is the
        # authoritative separation between state and treatment variables.
        try:
            manifest = parse_analysis_manifest(column_manifest)
            verified_manifest_hash = manifest.verify_digest(manifest_sha256 or "")
            manifest.validate_analysis_input(
                records,
                feature_names=feature_names,
                treatment_columns=treatment_columns,
                patient_id_column=patient_id_column,
                index_date_column=index_date_column,
                event_time_column=time_column,
                feature_available_time_column=feature_available_time_column,
            )
        except ManifestValidationError as exc:
            raise StateDiscoveryError(str(exc)) from exc

        treatment_set = {_normalise_feature_name(name) for name in treatment_columns}
        prohibited = [
            name
            for name in feature_names
            if _is_treatment_feature(name)
            or _normalise_feature_name(name) in treatment_set
        ]
        if prohibited:
            raise StateDiscoveryError(
                "treatment/action variables cannot define disease state: "
                + ", ".join(sorted(prohibited))
            )
        assert_no_future_information(
            records,
            event_time_column=time_column,
            feature_available_time_column=feature_available_time_column,
        )

        matrix = _feature_matrix(records, feature_names)
        patient_ids = _patient_cluster_ids(records, patient_id_column)
        scaler = StandardScaler()
        standardised = scaler.fit_transform(matrix)
        mixture = GaussianMixture(
            n_components=self.n_states,
            covariance_type="diag",
            n_init=10,
            max_iter=500,
            reg_covar=1e-6,
            random_state=self.random_seed,
        )
        raw_labels = mixture.fit_predict(standardised)

        # Canonicalise arbitrary mixture component numbers by the first
        # standardised centroid coordinate, then by the remaining coordinates.
        component_order = sorted(
            range(self.n_states), key=lambda i: tuple(mixture.means_[i].tolist())
        )
        remap = {old: new for new, old in enumerate(component_order)}
        labels = np.asarray([remap[int(label)] for label in raw_labels], dtype=int)
        probabilities_raw = mixture.predict_proba(standardised)
        probabilities = probabilities_raw[:, component_order]
        state_labels = tuple(f"state_{index + 1}" for index in range(self.n_states))

        original_centres = scaler.inverse_transform(mixture.means_[component_order])
        centroids = {
            state_labels[index]: {
                feature: float(original_centres[index, column])
                for column, feature in enumerate(feature_names)
            }
            for index in range(self.n_states)
        }
        counts, person_time, rates = self._transition_intensities(
            records,
            labels,
            patient_id_column=patient_id_column,
            time_column=time_column,
        )
        bootstrap_ari = self._bootstrap_stability(
            standardised, labels, patient_ids
        )

        candidate_states: list[CandidateDiseaseState] = []
        for index, label in enumerate(state_labels):
            state_rows = labels == index
            candidate = _safe_model(
                CandidateDiseaseState,
                {
                    "id": label,
                    "state_id": label,
                    "label": label,
                    "name": label,
                    "feature_centroid": centroids[label],
                    "centroid": centroids[label],
                    "feature_names": list(feature_names),
                    "patient_count": len(set(patient_ids[state_rows].tolist())),
                    "observation_count": int(np.sum(state_rows)),
                    "source": "interpretable_continuous_time_gmm",
                    "stability_ari": bootstrap_ari,
                    "research_only": True,
                    "eligible_for_guideline": False,
                },
            )
            if candidate is not None:
                candidate_states.append(candidate)

        state_transitions: list[StateTransition] = []
        for source in range(self.n_states):
            for target in range(self.n_states):
                if source == target or counts[source, target] == 0:
                    continue
                transition = _safe_model(
                    StateTransition,
                    {
                        "source_state_id": state_labels[source],
                        "from_state_id": state_labels[source],
                        "target_state_id": state_labels[target],
                        "to_state_id": state_labels[target],
                        "transition_count": int(counts[source, target]),
                        "count": int(counts[source, target]),
                        "person_time_days": float(person_time[source]),
                        "rate_per_day": float(rates[source, target]),
                        "transition_rate": float(rates[source, target]),
                        "treatment_conditioned": False,
                    },
                )
                if transition is not None:
                    state_transitions.append(transition)

        fitted = FittedStateModel(
            feature_names=tuple(feature_names),
            treatment_columns=tuple(treatment_columns),
            analysis_manifest_sha256=verified_manifest_hash,
            analysis_column_roles={
                name: role.value for name, role in manifest.role_by_name.items()
            },
            scaler=scaler,
            mixture=mixture,
            state_labels=state_labels,
            component_order=tuple(component_order),
            centroids=centroids,
            transition_rates=rates,
            transition_counts=counts,
            person_time=person_time,
            random_seed=self.random_seed,
            candidate_states=tuple(candidate_states),
            state_transitions=tuple(state_transitions),
        )
        return StateFitReport(
            model=fitted,
            labels=tuple(int(label) for label in labels),
            posterior_probabilities=tuple(
                tuple(float(value) for value in row) for row in probabilities
            ),
            bootstrap_ari=bootstrap_ari,
            analysis_manifest_sha256=verified_manifest_hash,
        )

    def _transition_intensities(
        self,
        records: Sequence[Mapping[str, Any]],
        labels: np.ndarray,
        *,
        patient_id_column: str,
        time_column: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        by_patient: defaultdict[str, list[tuple[datetime, int, int]]] = defaultdict(list)
        for row_number, (record, label) in enumerate(zip(records, labels, strict=True)):
            patient = record.get(patient_id_column)
            event_time = record.get(time_column)
            if patient is None or event_time is None:
                raise StateDiscoveryError(
                    f"record {row_number} requires {patient_id_column!r} and {time_column!r}"
                )
            by_patient[str(patient)].append((_as_datetime(event_time), int(label), row_number))

        counts = np.zeros((self.n_states, self.n_states), dtype=int)
        person_time = np.zeros(self.n_states, dtype=float)
        for observations in by_patient.values():
            observations.sort(key=lambda item: (item[0], item[2]))
            for (start, source, _), (end, target, _) in zip(
                observations, observations[1:]
            ):
                elapsed = (end - start).total_seconds() / 86_400
                if elapsed < 0:  # pragma: no cover - sorting protects this invariant
                    raise StateDiscoveryError("patient events are not chronological")
                elapsed = max(elapsed, self.minimum_elapsed_days)
                person_time[source] += elapsed
                if source != target:
                    counts[source, target] += 1
        rates = np.divide(
            counts,
            person_time[:, None],
            out=np.zeros_like(counts, dtype=float),
            where=person_time[:, None] > 0,
        )
        return counts, person_time, rates

    def _bootstrap_stability(
        self,
        matrix: np.ndarray,
        reference: np.ndarray,
        patient_ids: Sequence[str],
    ) -> float:
        """Estimate stability with patient-cluster bootstrap samples.

        Challengers train on all observations for each resampled patient.  ARI
        is always evaluated on the original fixed matrix, so replicate scores
        are directly comparable and repeated visits never become independent
        sampling units.
        """

        if len(patient_ids) != len(matrix):
            raise StateDiscoveryError(
                "patient cluster labels must align with the state feature matrix"
            )
        rng = np.random.default_rng(self.random_seed)
        scores: list[float] = []
        for replicate in range(self.bootstrap_replicates):
            indices = _patient_cluster_bootstrap_indices(patient_ids, rng)
            if len(np.unique(indices)) < self.n_states:
                continue
            challenger = clone(
                GaussianMixture(
                    n_components=self.n_states,
                    covariance_type="diag",
                    n_init=5,
                    max_iter=500,
                    reg_covar=1e-6,
                    random_state=self.random_seed + replicate + 1,
                )
            )
            challenger.fit(matrix[indices])
            scores.append(float(adjusted_rand_score(reference, challenger.predict(matrix))))
        return float(np.mean(scores)) if scores else 0.0


# Clear, discoverable aliases used by the CLI and external research notebooks.
StateDiscoveryEngine = ContinuousTimeStateDiscovery
temporal_patient_split = chronological_patient_split


__all__ = [
    "AnalysisColumn",
    "AnalysisColumnManifest",
    "ChallengerComparison",
    "ColumnRole",
    "ContinuousTimeStateDiscovery",
    "FittedStateModel",
    "PatientPartition",
    "StateDiscoveryEngine",
    "StateDiscoveryError",
    "StateFitReport",
    "StateModelChallenger",
    "StateValidationGate",
    "StateValidationThresholds",
    "assert_no_future_information",
    "chronological_patient_split",
    "canonical_manifest_sha256",
    "compare_challenger",
    "evaluate_state_validation_gate",
    "temporal_patient_split",
]
