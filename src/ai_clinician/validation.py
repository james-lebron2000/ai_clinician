"""Deterministic timeline validation metrics and the phase-one Go/No-Go gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from .models import (
    ClinicalEvent,
    PatientTimeline,
    TimelineGateThresholds,
    TimelineQualityReport,
)
from .timeline import leakage_reasons


UTC = timezone.utc
CRITICAL_CATEGORIES: tuple[str, ...] = ("treatment_line", "progression", "death")


@dataclass(frozen=True, slots=True)
class GoNoGoThresholds:
    minimum_total_gold_patients: int = 500
    minimum_gold_patients: int = 100
    minimum_event_precision: float = 0.90
    minimum_event_recall: float = 0.90
    minimum_treatment_line_macro_f1: float = 0.90
    minimum_date_within_14d: float = 0.90
    minimum_kappa: float = 0.80
    maximum_future_leakage_count: int = 0

    def __post_init__(self) -> None:
        if self.minimum_total_gold_patients < 500:
            raise ValueError("total gold cohort cannot weaken the 500-case gate")
        if self.minimum_gold_patients < 100:
            raise ValueError("locked reference set cannot weaken the 100-case gate")
        bounded = (
            self.minimum_event_precision,
            self.minimum_event_recall,
            self.minimum_treatment_line_macro_f1,
            self.minimum_date_within_14d,
            self.minimum_kappa,
        )
        if any(value < 0 or value > 1 for value in bounded):
            raise ValueError("quality thresholds must lie in [0, 1]")
        if any(value < 0.90 for value in bounded[:4]) or self.minimum_kappa < 0.80:
            raise ValueError("quality thresholds cannot be weaker than the protocol")
        if self.maximum_future_leakage_count != 0:
            raise ValueError("future information leakage threshold must remain zero")


def _event_type(event: ClinicalEvent) -> str:
    return str(getattr(event.event_type, "value", event.event_type))


def _event_category(event: ClinicalEvent) -> str | None:
    event_type = _event_type(event)
    if event_type in {"treatment_start", "treatment_end", "treatment"}:
        return "treatment_line" if _treatment_line(event) is not None else None
    if event_type in {"progression", "death"}:
        return event_type
    return None


def _treatment_line(event: ClinicalEvent) -> str | None:
    code = event.code or ""
    if code.startswith("treatment_line:"):
        return code.partition(":")[2]
    if isinstance(event.value, str):
        import re

        match = re.search(r"(?:第)?([一二三四五六七八九十\d]+)线", event.value)
        if match:
            return match.group(1)
    return None


def _identity(event: ClinicalEvent) -> tuple[str, str]:
    category = _event_category(event)
    if category is None:
        raise ValueError("event is not part of the critical validation set")
    label = _treatment_line(event) if category == "treatment_line" else category
    return category, label or "unknown"


def precision_recall_f1(
    gold_items: Sequence[Hashable], predicted_items: Sequence[Hashable]
) -> dict[str, float | int]:
    """Return multiset precision/recall/F1 with explicit zero-denominator rules."""

    gold = Counter(gold_items)
    predicted = Counter(predicted_items)
    true_positive = sum((gold & predicted).values())
    false_positive = sum((predicted - gold).values())
    false_negative = sum((gold - predicted).values())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else (1.0 if not gold else 0.0)
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def classification_metrics(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    *,
    labels: Sequence[Hashable] | None = None,
) -> dict[Hashable, dict[str, float | int]]:
    """Calculate one-vs-rest metrics for aligned discrete labels."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    selected = tuple(labels) if labels is not None else tuple(
        sorted(set(y_true) | set(y_pred), key=str)
    )
    metrics: dict[Hashable, dict[str, float | int]] = {}
    for label in selected:
        true_positive = sum(
            gold == label and predicted == label
            for gold, predicted in zip(y_true, y_pred)
        )
        false_positive = sum(
            gold != label and predicted == label
            for gold, predicted in zip(y_true, y_pred)
        )
        false_negative = sum(
            gold == label and predicted != label
            for gold, predicted in zip(y_true, y_pred)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def macro_f1(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    *,
    labels: Sequence[Hashable] | None = None,
) -> float:
    per_label = classification_metrics(y_true, y_pred, labels=labels)
    if not per_label:
        return 0.0
    return sum(float(metric["f1"]) for metric in per_label.values()) / len(per_label)


def cohen_kappa(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> float:
    """Compute unweighted Cohen's kappa without an optional ML dependency."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return 0.0
    total = len(y_true)
    observed = sum(gold == predicted for gold, predicted in zip(y_true, y_pred)) / total
    gold_counts = Counter(y_true)
    predicted_counts = Counter(y_pred)
    expected = sum(
        (gold_counts[label] / total) * (predicted_counts[label] / total)
        for label in set(gold_counts) | set(predicted_counts)
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _as_utc(value: datetime | date | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace("Z", "+00:00")
        if not cleaned:
            return None
        try:
            value = datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def date_within_tolerance(
    predicted_dates: Sequence[datetime | date | None],
    gold_dates: Sequence[datetime | date | None],
    *,
    tolerance_days: int = 14,
) -> float:
    """Proportion of aligned gold dates recovered within a calendar tolerance."""

    if len(predicted_dates) != len(gold_dates):
        raise ValueError("predicted_dates and gold_dates must have equal length")
    if tolerance_days < 0:
        raise ValueError("tolerance_days cannot be negative")
    if not gold_dates:
        return 0.0
    passed = 0
    for predicted, gold in zip(predicted_dates, gold_dates):
        predicted_at = _as_utc(predicted)
        gold_at = _as_utc(gold)
        if predicted_at is None or gold_at is None:
            continue
        if abs((predicted_at.date() - gold_at.date()).days) <= tolerance_days:
            passed += 1
    return passed / len(gold_dates)


def timeline_integrity_issues(
    timeline: PatientTimeline,
    *,
    decision_time: datetime | date | None = None,
) -> tuple[str, ...]:
    """Check provenance, patient linkage, bounds and optional temporal leakage."""

    issues: list[str] = []
    seen_ids: set[str] = set()
    for event in timeline.events:
        if event.id in seen_ids:
            issues.append(f"duplicate_event_id:{event.id}")
        seen_ids.add(event.id)
        if event.patient_key != timeline.patient_key:
            issues.append(f"patient_key_mismatch:{event.id}")
        if not event.evidence_spans:
            issues.append(f"missing_evidence:{event.id}")
        if not 0 <= event.confidence <= 1:
            issues.append(f"confidence_out_of_bounds:{event.id}")
        for span in event.evidence_spans:
            if span.start < 0 or span.end <= span.start:
                issues.append(f"invalid_evidence_span:{event.id}")
            if len(span.text_sha256) != 64:
                issues.append(f"invalid_evidence_hash:{event.id}")
    issues.extend(timeline.leakage_flags)
    if decision_time is not None:
        issues.extend(leakage_reasons(timeline, decision_time))
    return tuple(sorted(set(issues)))


def _critical_events(timeline: PatientTimeline, category: str) -> list[ClinicalEvent]:
    return [event for event in timeline.events if _event_category(event) == category]


def _event_time_sort(event: ClinicalEvent) -> tuple[datetime, str]:
    maximum = datetime.max.replace(tzinfo=UTC)
    return _as_utc(event.event_time) or maximum, event.id


def _aligned_treatment_labels(
    predicted: Mapping[str, PatientTimeline], gold: Mapping[str, PatientTimeline]
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for patient_key in sorted(set(predicted) | set(gold)):
        gold_events = sorted(
            _critical_events(gold[patient_key], "treatment_line") if patient_key in gold else [],
            key=_event_time_sort,
        )
        predicted_events = sorted(
            _critical_events(predicted[patient_key], "treatment_line")
            if patient_key in predicted
            else [],
            key=_event_time_sort,
        )
        length = max(len(gold_events), len(predicted_events))
        for index in range(length):
            y_true.append(
                _treatment_line(gold_events[index])
                if index < len(gold_events)
                else "__spurious__"
            )
            y_pred.append(
                _treatment_line(predicted_events[index])
                if index < len(predicted_events)
                else "__missing__"
            )
    return y_true, y_pred


def _date_pairs(
    predicted: Mapping[str, PatientTimeline], gold: Mapping[str, PatientTimeline]
) -> tuple[list[datetime | None], list[datetime | None]]:
    """Greedily align critical events by identity, then nearest date."""

    predicted_dates: list[datetime | None] = []
    gold_dates: list[datetime | None] = []
    for patient_key in sorted(set(gold) | set(predicted)):
        gold_events = gold.get(patient_key)
        predicted_events = predicted.get(patient_key)
        by_identity_gold: dict[tuple[str, str], list[ClinicalEvent]] = defaultdict(list)
        by_identity_predicted: dict[tuple[str, str], list[ClinicalEvent]] = defaultdict(list)
        if gold_events:
            for event in gold_events.events:
                if _event_category(event) is not None:
                    by_identity_gold[_identity(event)].append(event)
        if predicted_events:
            for event in predicted_events.events:
                if _event_category(event) is not None:
                    by_identity_predicted[_identity(event)].append(event)

        for identity, expected_events in by_identity_gold.items():
            remaining = list(by_identity_predicted.get(identity, []))
            for expected in sorted(expected_events, key=_event_time_sort):
                expected_at = _as_utc(expected.event_time)
                if not remaining:
                    predicted_dates.append(None)
                    gold_dates.append(expected_at)
                    continue
                if expected_at is None:
                    chosen_index = 0
                else:
                    chosen_index = min(
                        range(len(remaining)),
                        key=lambda index: (
                            abs(
                                (
                                    (_as_utc(remaining[index].event_time) or datetime.max.replace(tzinfo=UTC))
                                    - expected_at
                                ).total_seconds()
                            ),
                            remaining[index].id,
                        ),
                    )
                chosen = remaining.pop(chosen_index)
                predicted_dates.append(_as_utc(chosen.event_time))
                gold_dates.append(expected_at)
    return predicted_dates, gold_dates


def evaluate_gold_standard(
    predicted_timelines: Sequence[PatientTimeline],
    gold_timelines: Sequence[PatientTimeline],
    *,
    decision_views: Sequence[Mapping[str, Any]] | None = None,
    require_decision_views: bool = True,
    gold_snapshot_id: str | None = None,
    gold_snapshot_sha256: str | None = None,
    locked_split_sha256: str | None = None,
    locked_fraction: float | None = None,
    decision_view_manifest_sha256: str | None = None,
    extractor_release_id: str | None = None,
    extractor_sha256: str | None = None,
    evaluation_code_sha256: str | None = None,
    gold_total_case_count: int | None = None,
    annotation_kappa: float | None = None,
    manifest_registration_id: str | None = None,
    manifest_registration_hash: str | None = None,
    annotation_review_ids: Sequence[str] = (),
    thresholds: GoNoGoThresholds = GoNoGoThresholds(),
) -> TimelineQualityReport:
    """Evaluate the locked gold standard and apply the phase-one hard gate."""

    predicted = {timeline.patient_key: timeline for timeline in predicted_timelines}
    gold = {timeline.patient_key: timeline for timeline in gold_timelines}
    if len(predicted) != len(predicted_timelines):
        raise ValueError("predicted timelines contain duplicate patient keys")
    if len(gold) != len(gold_timelines):
        raise ValueError("gold timelines contain duplicate patient keys")

    counts: dict[str, int] = {
        "gold_patient_count": len(gold),
        "predicted_patient_count": len(predicted),
    }
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    failures: list[str] = []
    required_patient_keys = set(predicted) | set(gold)
    audited_patient_keys: set[str] = set()
    decision_view_failures: list[str] = []
    seen_decisions: set[tuple[str, str]] = set()
    for index, view in enumerate(decision_views or ()):
        patient_key = str(view.get("patient_key", "")).strip()
        decision_point_id = str(view.get("decision_point_id", "")).strip()
        feature_event_ids = view.get("feature_event_ids")
        decision_time = _as_utc(view.get("decision_time"))
        if (
            not patient_key
            or not decision_point_id
            or decision_time is None
            or not isinstance(feature_event_ids, Sequence)
            or isinstance(feature_event_ids, (str, bytes))
            or not feature_event_ids
        ):
            decision_view_failures.append(f"invalid_decision_view:{index}")
            continue
        decision_key = (patient_key, decision_point_id)
        if decision_key in seen_decisions:
            decision_view_failures.append(f"duplicate_decision_view:{index}")
            continue
        seen_decisions.add(decision_key)
        audited_patient_keys.add(patient_key)
        timeline = predicted.get(patient_key)
        if timeline is None:
            decision_view_failures.append(f"missing_predicted_timeline:{index}")
            continue
        events = {event.id: event for event in timeline.events}
        event_ids = [str(item) for item in feature_event_ids]
        if len(event_ids) != len(set(event_ids)):
            decision_view_failures.append(f"duplicate_feature_event:{index}")
        for event_id in event_ids:
            event = events.get(event_id)
            if event is None:
                decision_view_failures.append(f"unknown_feature_event:{index}")
                continue
            event_time = _as_utc(event.event_time)
            available_at = _as_utc(event.source_available_at)
            if event_time is None or event_time > decision_time:
                decision_view_failures.append(f"future_or_unknown_event:{index}")
            if available_at is None or available_at > decision_time:
                decision_view_failures.append(
                    f"future_or_unknown_source_availability:{index}"
                )
    missing_audits = required_patient_keys - audited_patient_keys
    if require_decision_views and missing_audits:
        decision_view_failures.append("decision_views_missing_for_patient_union")
    if decision_view_failures:
        failures.append("decision_view_leakage_audit_failed")
    effective_gold_total = (
        int(gold_total_case_count)
        if gold_total_case_count is not None
        else len(gold)
    )
    if effective_gold_total < thresholds.minimum_total_gold_patients:
        failures.append("total_gold_patient_count_below_threshold")
    if len(gold) < thresholds.minimum_gold_patients:
        failures.append("gold_patient_count_below_threshold")
    if len(gold) * 5 != effective_gold_total:
        failures.append("locked_reference_is_not_exactly_twenty_percent")
    for category in CRITICAL_CATEGORIES:
        gold_items: list[tuple[str, str]] = []
        predicted_items: list[tuple[str, str]] = []
        for patient_key, timeline in gold.items():
            gold_items.extend(
                (patient_key, _identity(event)[1])
                for event in _critical_events(timeline, category)
            )
        for patient_key, timeline in predicted.items():
            predicted_items.extend(
                (patient_key, _identity(event)[1])
                for event in _critical_events(timeline, category)
            )
        metric = precision_recall_f1(gold_items, predicted_items)
        counts[f"gold_{category}"] = len(gold_items)
        counts[f"predicted_{category}"] = len(predicted_items)
        counts[f"tp_{category}"] = int(metric["tp"])
        counts[f"fp_{category}"] = int(metric["fp"])
        counts[f"fn_{category}"] = int(metric["fn"])
        precision[category] = float(metric["precision"])
        recall[category] = float(metric["recall"])
        f1[category] = float(metric["f1"])
        if not gold_items:
            failures.append(f"insufficient_gold_events:{category}")
        if precision[category] < thresholds.minimum_event_precision:
            failures.append(f"precision_below_threshold:{category}")
        if recall[category] < thresholds.minimum_event_recall:
            failures.append(f"recall_below_threshold:{category}")

    y_true, y_pred = _aligned_treatment_labels(predicted, gold)
    line_labels = sorted(
        (set(y_true) | set(y_pred)) - {"__missing__", "__spurious__"}, key=str
    )
    line_macro_f1 = macro_f1(y_true, y_pred, labels=line_labels) if line_labels else 0.0
    model_gold_kappa = cohen_kappa(y_true, y_pred)
    kappa = float(annotation_kappa) if annotation_kappa is not None else -1.0
    predicted_dates, gold_dates = _date_pairs(predicted, gold)
    date_score = date_within_tolerance(predicted_dates, gold_dates, tolerance_days=14)

    future_leakage_count = sum(
        len(timeline.leakage_flags) for timeline in predicted.values()
    ) + len(decision_view_failures)

    if line_macro_f1 < thresholds.minimum_treatment_line_macro_f1:
        failures.append("treatment_line_macro_f1_below_threshold")
    if date_score < thresholds.minimum_date_within_14d:
        failures.append("date_within_14d_below_threshold")
    if kappa < thresholds.minimum_kappa:
        failures.append("annotation_cohen_kappa_below_threshold")
    if future_leakage_count > thresholds.maximum_future_leakage_count:
        failures.append("future_information_leakage_detected")
    provenance_values = (
        gold_snapshot_id,
        gold_snapshot_sha256,
        locked_split_sha256,
        locked_fraction == 0.20,
        decision_view_manifest_sha256,
        extractor_release_id,
        extractor_sha256,
        evaluation_code_sha256,
        manifest_registration_id,
        manifest_registration_hash,
        len(set(annotation_review_ids)) >= 3,
    )
    if not all(provenance_values):
        failures.append("evaluation_provenance_incomplete")

    # The public report stores the worst critical category.  This preserves the
    # per-category hard gate: an excellent death extractor cannot mask weak
    # progression or treatment-line performance in a pooled average.
    return TimelineQualityReport(
        gold_total_case_count=effective_gold_total,
        reference_case_count=len(gold),
        critical_event_counts={
            category: counts[f"gold_{category}"] for category in CRITICAL_CATEGORIES
        },
        decision_view_patient_count=len(audited_patient_keys),
        decision_view_count=len(seen_decisions),
        decision_view_failure_count=len(decision_view_failures),
        critical_event_precision=min(precision.values(), default=0.0),
        critical_event_recall=min(recall.values(), default=0.0),
        critical_event_f1=min(f1.values(), default=0.0),
        treatment_line_macro_f1=line_macro_f1,
        date_within_14_days=date_score,
        cohen_kappa=kappa,
        model_gold_kappa=model_gold_kappa,
        future_leakage_count=future_leakage_count,
        thresholds=TimelineGateThresholds(
            minimum_gold_total_cases=thresholds.minimum_total_gold_patients,
            minimum_reference_cases=thresholds.minimum_gold_patients,
            minimum_event_precision=thresholds.minimum_event_precision,
            minimum_event_recall=thresholds.minimum_event_recall,
            minimum_critical_event_f1=0.90,
            minimum_treatment_line_macro_f1=(
                thresholds.minimum_treatment_line_macro_f1
            ),
            minimum_date_within_14_days=thresholds.minimum_date_within_14d,
            minimum_cohen_kappa=thresholds.minimum_kappa,
            maximum_future_leakage_count=thresholds.maximum_future_leakage_count,
        ),
        gold_snapshot_id=gold_snapshot_id,
        gold_snapshot_sha256=gold_snapshot_sha256,
        locked_split_sha256=locked_split_sha256,
        locked_fraction=locked_fraction,
        decision_view_manifest_sha256=decision_view_manifest_sha256,
        extractor_release_id=extractor_release_id,
        extractor_sha256=extractor_sha256,
        evaluation_code_sha256=evaluation_code_sha256,
        manifest_registration_id=manifest_registration_id,
        manifest_registration_hash=manifest_registration_hash,
        annotation_review_ids=list(annotation_review_ids),
        passed=not failures,
        failures=sorted(set(failures)),
    )


validate_timeline = timeline_integrity_issues
validate_gold_standard = evaluate_gold_standard
