from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ai_clinician.interop import (
    FHIR_EVIDENCE_SPAN_URL,
    FORBIDDEN_ORDER_RESOURCE_TYPES,
    clinical_event_to_fhir,
    clinical_event_to_omop,
    patient_timeline_to_fhir_bundle,
)
from ai_clinician.models import ClinicalEvent, EvidenceSpan, EventType, PatientTimeline


UTC = timezone.utc
PATIENT_KEY = "synthetic-patient-local-001"
SOURCE_DOCUMENT = "synthetic-secret-source-file.xlsx"
SOURCE_SHEET = "合成患者工作表"
RAW_EXCERPT = "合成原始病历摘录，绝对不能进入交换结果"


def _span(seed: int = 0) -> EvidenceSpan:
    return EvidenceSpan(
        source_document_id=SOURCE_DOCUMENT,
        sheet_name=SOURCE_SHEET,
        cell_ref=f"B{seed + 2}",
        start=seed,
        end=seed + 8,
        text_sha256=f"{seed + 1:064x}",
        excerpt=RAW_EXCERPT,
    )


def _event(
    event_type: EventType,
    *,
    seed: int = 0,
    code: str | None = None,
    value: str | int | float | bool | None = None,
    unit: str | None = None,
    with_time: bool = True,
) -> ClinicalEvent:
    return ClinicalEvent(
        id=f"synthetic-event-{seed}",
        patient_key=PATIENT_KEY,
        event_type=event_type,
        event_time=(
            datetime(2024, 1, min(seed + 1, 28), 9, tzinfo=UTC)
            if with_time
            else None
        ),
        time_precision="day" if with_time else "unknown",
        code=code,
        value=value,
        unit=unit,
        confidence=0.95,
        evidence_spans=[_span(seed)],
    )


def _all_events() -> list[ClinicalEvent]:
    definitions = [
        (EventType.DIAGNOSIS, "CRC", None, None),
        (EventType.METASTASIS, "liver-metastasis", None, None),
        (EventType.PROGRESSION, "radiographic-progression", None, None),
        (EventType.ECOG, "ECOG", 1, None),
        (EventType.PATHOLOGY_STAGE, "TNM", "T3N1M1", None),
        (EventType.MOLECULAR_TEST, "KRAS", "mutated", None),
        (EventType.LABORATORY, "CEA", 12.5, "ng/mL"),
        (EventType.RESPONSE, "RECIST", "PR", None),
        (EventType.TOXICITY, "CTCAE", 3, "grade"),
        (EventType.SURGERY, "colectomy", None, None),
        (EventType.TREATMENT_START, "FOLFOX", None, None),
        (EventType.TREATMENT_END, "FOLFOX", None, None),
        (EventType.FOLLOW_UP, "oncology-follow-up", None, None),
        (EventType.DEATH, "death", None, None),
    ]
    return [
        _event(kind, seed=index, code=code, value=value, unit=unit)
        for index, (kind, code, value, unit) in enumerate(definitions)
    ]


@pytest.mark.parametrize(
    ("event_type", "domain", "table"),
    [
        (EventType.DIAGNOSIS, "Condition", "condition_occurrence"),
        (EventType.METASTASIS, "Condition", "condition_occurrence"),
        (EventType.PROGRESSION, "Condition", "condition_occurrence"),
        (EventType.ECOG, "Observation", "observation"),
        (EventType.PATHOLOGY_STAGE, "Observation", "observation"),
        (EventType.MOLECULAR_TEST, "Observation", "observation"),
        (EventType.LABORATORY, "Measurement", "measurement"),
        (EventType.SURGERY, "Procedure", "procedure_occurrence"),
        (EventType.TREATMENT_START, "Drug", "drug_exposure"),
        (EventType.TREATMENT_END, "Drug", "drug_exposure"),
        (EventType.RESPONSE, "Observation", "observation"),
        (EventType.TOXICITY, "Observation", "observation"),
        (EventType.FOLLOW_UP, "Visit", "visit_occurrence"),
        (EventType.DEATH, "Death", "death"),
    ],
)
def test_event_to_omop_maps_every_domain_and_keeps_key_local(
    event_type: EventType, domain: str, table: str
) -> None:
    event = _event(event_type, code="synthetic-code")
    record = clinical_event_to_omop(event)

    assert record.domain == domain
    assert record.table == table
    assert record.patient_key_local == PATIENT_KEY
    assert record.event_date == datetime(2024, 1, 1, tzinfo=UTC).date()
    assert record.event_datetime == datetime(2024, 1, 1, 9, tzinfo=UTC)
    assert record.source_value == "synthetic-code"
    assert record.concept_id is None


def test_omop_temporal_roles_and_reviewed_concept_mapping() -> None:
    start = clinical_event_to_omop(
        _event(EventType.TREATMENT_START, code="FOLFOX"), concept_id=123
    )
    end = clinical_event_to_omop(
        _event(EventType.TREATMENT_END, code="FOLFOX")
    )

    assert start.temporal_role == "start"
    assert start.start_datetime is not None
    assert start.end_datetime is None
    assert start.concept_id == 123
    assert end.temporal_role == "end"
    assert end.start_datetime is None
    assert end.end_datetime is not None

    with pytest.raises(ValidationError):
        clinical_event_to_omop(
            _event(EventType.DIAGNOSIS, code="CRC"), concept_id=-1
        )


def test_fhir_bundle_maps_all_events_without_orders_or_raw_evidence() -> None:
    timeline = PatientTimeline(
        patient_key=PATIENT_KEY,
        events=_all_events(),
        completeness=1.0,
    )

    bundle = patient_timeline_to_fhir_bundle(timeline)
    resources = [entry["resource"] for entry in bundle["entry"]]
    resource_types = [resource["resourceType"] for resource in resources]

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert set(resource_types) == {
        "Condition",
        "Observation",
        "Procedure",
        "MedicationAdministration",
        "Encounter",
    }
    assert resource_types.count("MedicationAdministration") == 2
    assert not (set(resource_types) & FORBIDDEN_ORDER_RESOURCE_TYPES)
    assert "Patient" not in resource_types

    serialized = json.dumps(bundle, ensure_ascii=False, sort_keys=True)
    assert PATIENT_KEY not in serialized
    assert SOURCE_DOCUMENT not in serialized
    assert SOURCE_SHEET not in serialized
    assert RAW_EXCERPT not in serialized
    assert "excerpt" not in serialized.lower()
    for forbidden in FORBIDDEN_ORDER_RESOURCE_TYPES:
        assert f'"resourceType": "{forbidden}"' not in serialized

    for resource in resources:
        assert resource["subject"]["reference"].startswith("Patient/subject-")
        evidence = [
            extension
            for extension in resource["extension"]
            if extension["url"] == FHIR_EVIDENCE_SPAN_URL
        ]
        assert len(evidence) == 1
        locator_urls = {item["url"] for item in evidence[0]["extension"]}
        assert {
            "source-document-sha256",
            "sheet-name-sha256",
            "cell-reference",
            "start-offset",
            "end-offset",
            "text-sha256",
        } <= locator_urls


def test_fhir_export_is_deterministic_and_handles_unknown_medication_time() -> None:
    event = _event(
        EventType.TREATMENT_START,
        code="FOLFOX",
        with_time=False,
    )
    timeline = PatientTimeline(
        patient_key=PATIENT_KEY,
        events=[event],
        completeness=0.5,
    )

    first = patient_timeline_to_fhir_bundle(timeline)
    second = patient_timeline_to_fhir_bundle(timeline)
    assert first == second

    medication = first["entry"][0]["resource"]
    assert medication["resourceType"] == "MedicationAdministration"
    assert medication["effectiveDateTime"] is None
    assert medication["_effectiveDateTime"]["extension"][0]["valueCode"] == "unknown"


def test_single_event_fhir_adapter_never_returns_a_request_resource() -> None:
    for event in _all_events():
        resource = clinical_event_to_fhir(event)
        assert resource["resourceType"] not in FORBIDDEN_ORDER_RESOURCE_TYPES
