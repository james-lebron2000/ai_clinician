from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from ai_clinician.timeline import (
    HybridTimelineExtractor,
    build_timeline,
    decision_time_view,
    leakage_reasons,
)
from ai_clinician.validation import (
    GoNoGoThresholds,
    classification_metrics,
    cohen_kappa,
    date_within_tolerance,
    evaluate_gold_standard,
)


UTC = timezone.utc


def _event_type(event):
    return str(getattr(event.event_type, "value", event.event_type))


def _synthetic_timeline():
    text = (
        "2022年1月3日确诊直肠癌，ECOG评分1分。"
        "2022-02-01开始一线mFOLFOX6联合贝伐珠单抗。"
        "2022年6月15日影像学提示疾病进展并出现肝转移。"
        "2022年7月1日改用二线FOLFIRI，发生3级中性粒细胞减少。"
        "2023年1月1日患者死亡。"
        "未见肺转移，无4级腹泻。"
    )
    return text, build_timeline(
        "SYNTHETIC-PATIENT-001",
        text,
        source_document_id="synthetic-note-001",
        recorded_at=datetime(2022, 1, 4, tzinfo=UTC),
        sheet_name="合成病例",
        cell_ref="A1",
    )


def _locked_gold_cohort(case_count=500):
    _, template = _synthetic_timeline()
    timelines = []
    decision_views = []
    for index in range(case_count):
        patient_key = f"SYNTHETIC-GOLD-{index:04d}"
        events = [
            event.model_copy(
                update={"id": f"{event.id}-{index:04d}", "patient_key": patient_key}
            )
            for event in template.events
        ]
        timeline = template.model_copy(
            update={"patient_key": patient_key, "events": events}
        )
        timelines.append(timeline)
        decision_views.append(
            {
                "patient_key": patient_key,
                "decision_point_id": f"decision-{index:04d}",
                "decision_time": datetime(2024, 1, 1, tzinfo=UTC),
                "feature_event_ids": [event.id for event in events],
            }
        )
    return timelines, decision_views


def test_deterministic_chinese_rule_extraction_with_provenance_and_negation():
    text, timeline = _synthetic_timeline()
    types = [_event_type(event) for event in timeline.events]

    assert types.count("treatment_start") == 2
    assert types.count("progression") == 1
    assert types.count("metastasis") == 1  # negated lung metastasis is excluded
    assert types.count("death") == 1
    assert types.count("ecog") == 1
    assert types.count("toxicity") == 1  # negated grade-4 diarrhoea is excluded
    treatment = next(event for event in timeline.events if event.code == "treatment_line:1")
    assert treatment.value == "FOLFOX+贝伐珠单抗"
    toxicity = next(event for event in timeline.events if _event_type(event) == "toxicity")
    assert toxicity.value == 3
    assert toxicity.unit == "CTCAE grade"
    assert timeline.completeness == 1.0

    for event in timeline.events:
        assert event.evidence_spans
        span = event.evidence_spans[0]
        assert span.source_document_id == "synthetic-note-001"
        assert span.sheet_name == "合成病例"
        assert span.cell_ref == "A1"
        assert span.start < span.end
        assert span.text_sha256 == hashlib.sha256(
            text[span.start : span.end].encode("utf-8")
        ).hexdigest()
        assert 0 <= event.confidence <= 1
        assert event.time_precision == "day"

    _, repeated = _synthetic_timeline()
    assert [event.id for event in repeated.events] == [event.id for event in timeline.events]


def test_conflicts_and_missing_temporal_metadata_are_never_silently_removed():
    conflicting = build_timeline(
        "SYNTHETIC-PATIENT-002",
        "2022年1月ECOG评分1分。2022年1月ECOG评分2分。",
        source_document_id="synthetic-note-002",
        recorded_at=datetime(2022, 2, 1, tzinfo=UTC),
    )
    ecog = [event for event in conflicting.events if _event_type(event) == "ecog"]
    assert len(ecog) == 2
    assert all(event.time_precision == "month" for event in ecog)
    assert all("conflicting_values_same_time" in event.conflicts for event in ecog)
    assert conflicting.conflicts

    missing = build_timeline(
        "SYNTHETIC-PATIENT-003",
        "影像学确认疾病进展。",
        source_document_id="synthetic-note-003",
    )
    event = missing.events[0]
    assert event.event_time is None
    assert event.time_precision == "unknown"
    assert "event_time_missing" in event.missing_flags
    assert "source_available_at_missing" in event.missing_flags


def test_multiple_lines_in_one_clause_and_coexisting_metastases_are_distinct():
    timeline = build_timeline(
        "SYNTHETIC-PATIENT-006",
        (
            "2022年1月1日停用一线FOLFOX，"
            "2022年2月1日开始二线FOLFIRI并确认肝转移、肺转移，未行三线治疗。"
        ),
        source_document_id="synthetic-note-006",
        recorded_at=datetime(2022, 2, 2, tzinfo=UTC),
    )

    lines = {
        event.code: _event_type(event)
        for event in timeline.events
        if event.code and event.code.startswith("treatment_line:")
    }
    assert lines == {
        "treatment_line:1": "treatment_end",
        "treatment_line:2": "treatment_start",
    }
    metastases = [event for event in timeline.events if _event_type(event) == "metastasis"]
    assert {event.value for event in metastases} == {"肝", "肺"}
    assert all(not event.conflicts for event in metastases)


def test_decision_time_view_excludes_future_and_unknown_evidence():
    _, timeline = _synthetic_timeline()
    cutoff = datetime(2022, 3, 1, tzinfo=UTC)

    view = decision_time_view(timeline, cutoff)

    assert {_event_type(event) for event in view.events} == {
        "diagnosis",
        "ecog",
        "treatment_start",
    }
    assert not leakage_reasons(view, cutoff)
    reasons = leakage_reasons(timeline, cutoff)
    assert any(reason.startswith("future_event_time:") for reason in reasons)

    unknown = build_timeline(
        "SYNTHETIC-PATIENT-004",
        "2022年1月1日疾病进展。",
        source_document_id="synthetic-note-004",
    )
    assert decision_time_view(unknown, cutoff).events == []
    assert leakage_reasons(unknown, cutoff) == (
        f"source_availability_unknown:{unknown.events[0].id}",
    )


def test_only_explicitly_local_pluggable_extractors_are_accepted():
    class EmptyLocalExtractor:
        execution_mode = "local"

        def extract(self, **kwargs):
            assert kwargs["text"] == "2022年1月1日疾病进展。"
            return []

    extractor = HybridTimelineExtractor(EmptyLocalExtractor())
    timeline = extractor.extract(
        "SYNTHETIC-PATIENT-005",
        "2022年1月1日疾病进展。",
        source_document_id="synthetic-note-005",
        recorded_at=datetime(2022, 1, 2, tzinfo=UTC),
    )
    assert len(timeline.events) == 1

    class RemoteExtractor:
        execution_mode = "remote"

    with pytest.raises(ValueError, match="execution_mode='local'"):
        HybridTimelineExtractor(RemoteExtractor())


def test_local_extractor_cannot_forge_evidence_or_availability():
    class ForgedLocalExtractor:
        execution_mode = "local"

        def extract(self, **kwargs):
            return [
                {
                    "patient_key": kwargs["patient_key"],
                    "event_type": "progression",
                    "confidence": 0.9,
                    "source_available_at": kwargs["recorded_at"],
                    "evidence_spans": [
                        {
                            "source_document_id": kwargs["source_document_id"],
                            "start": 0,
                            "end": 4,
                            "text_sha256": "0" * 64,
                        }
                    ],
                }
            ]

    with pytest.raises(ValueError, match="evidence hash"):
        HybridTimelineExtractor(ForgedLocalExtractor()).extract(
            "SYNTHETIC-PATIENT-006",
            "疾病进展。",
            source_document_id="synthetic-note-006",
            recorded_at=datetime(2022, 1, 2, tzinfo=UTC),
        )

    text = "疾病进展。"

    class WrongLocationExtractor:
        execution_mode = "local"

        def extract(self, **kwargs):
            return [
                {
                    "patient_key": kwargs["patient_key"],
                    "event_type": "progression",
                    "confidence": 0.9,
                    "source_available_at": kwargs["recorded_at"],
                    "evidence_spans": [
                        {
                            "source_document_id": kwargs["source_document_id"],
                            "sheet_name": "another-sheet",
                            "cell_ref": "C9",
                            "start": 0,
                            "end": 4,
                            "text_sha256": hashlib.sha256(
                                text[:4].encode("utf-8")
                            ).hexdigest(),
                        }
                    ],
                }
            ]

    with pytest.raises(ValueError, match="source location"):
        HybridTimelineExtractor(WrongLocationExtractor()).extract(
            "SYNTHETIC-PATIENT-006",
            text,
            source_document_id="synthetic-note-006",
            recorded_at=datetime(2022, 1, 2, tzinfo=UTC),
            sheet_name="expected-sheet",
            cell_ref="B2",
        )


def test_validation_metrics_and_go_no_go_gate():
    cohort, decision_views = _locked_gold_cohort()
    passed = evaluate_gold_standard(
        cohort[:100],
        cohort[:100],
        decision_views=decision_views[:100],
        gold_total_case_count=500,
        annotation_kappa=1.0,
        gold_snapshot_id="synthetic-gold-snapshot",
        gold_snapshot_sha256="a" * 64,
        locked_split_sha256="b" * 64,
        locked_fraction=0.20,
        decision_view_manifest_sha256="c" * 64,
        extractor_release_id="synthetic-extractor-release",
        extractor_sha256="d" * 64,
        evaluation_code_sha256="e" * 64,
        manifest_registration_id="synthetic-manifest-registration",
        manifest_registration_hash="f" * 64,
        annotation_review_ids=["review-1", "review-2", "review-3"],
    )

    assert passed.passed
    assert passed.critical_event_precision == 1.0
    assert passed.critical_event_recall == 1.0
    assert passed.critical_event_f1 == 1.0
    assert passed.treatment_line_macro_f1 == 1.0
    assert passed.date_within_14_days == 1.0
    assert passed.cohen_kappa == 1.0
    assert passed.future_leakage_count == 0

    _, gold = _synthetic_timeline()
    altered_events = []
    for event in gold.events:
        if _event_type(event) == "progression":
            continue
        if _event_type(event) == "death":
            event = event.model_copy(update={"event_time": event.event_time + timedelta(days=30)})
        if event.code == "treatment_line:2":
            event = event.model_copy(update={"code": "treatment_line:3"})
        altered_events.append(event)
    predicted = gold.model_copy(update={"events": altered_events})
    failed = evaluate_gold_standard(
        [predicted],
        [gold],
        decision_views=[
            {
                "patient_key": gold.patient_key,
                "decision_point_id": "decision-failed",
                "decision_time": datetime(2024, 1, 1, tzinfo=UTC),
                "feature_event_ids": [event.id for event in predicted.events],
            }
        ],
        gold_total_case_count=500,
        annotation_kappa=1.0,
    )

    assert not failed.passed
    assert failed.critical_event_recall == 0.0
    assert "recall_below_threshold:progression" in failed.failures
    assert "treatment_line_macro_f1_below_threshold" in failed.failures
    assert "date_within_14d_below_threshold" in failed.failures

    sample_too_small = evaluate_gold_standard([gold], [gold])
    assert not sample_too_small.passed
    assert "gold_patient_count_below_threshold" in sample_too_small.failures


def test_production_validation_requires_decision_views_for_patient_union():
    _, gold = _synthetic_timeline()
    report = evaluate_gold_standard(
        [gold],
        [gold],
        require_decision_views=True,
    )
    assert report.passed is False
    assert "decision_view_leakage_audit_failed" in report.failures


def test_standalone_metric_edge_cases_are_explicit():
    metrics = classification_metrics(["1", "2"], ["2", "1"])
    assert metrics["1"]["f1"] == 0.0
    assert metrics["2"]["f1"] == 0.0
    assert cohen_kappa(["1", "2"], ["1", "2"]) == 1.0
    assert date_within_tolerance(
        [datetime(2022, 1, 15, tzinfo=UTC), None],
        [datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 2, 1, tzinfo=UTC)],
    ) == 0.5
