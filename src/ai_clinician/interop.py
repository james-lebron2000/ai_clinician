"""Minimal, non-ordering OMOP-style and FHIR R4 interoperability.

The adapters in this module are deliberately one way: they serialize the
research event layer for analysis or review and never create a clinical
request/order resource.  Patient keys remain local to the OMOP-style record;
FHIR subject references use a deterministic opaque digest instead.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import Field

from .models import ClinicalEvent, EvidenceSpan, EventType, PatientTimeline, StrictModel


FHIR_BASE_URL = "https://ai-clinician.local/fhir"
FHIR_EVENT_CODE_SYSTEM = f"{FHIR_BASE_URL}/CodeSystem/clinical-event-type"
FHIR_EVIDENCE_SPAN_URL = f"{FHIR_BASE_URL}/StructureDefinition/evidence-span"
FHIR_TIME_PRECISION_URL = f"{FHIR_BASE_URL}/StructureDefinition/time-precision"
FHIR_CONFIDENCE_URL = f"{FHIR_BASE_URL}/StructureDefinition/extraction-confidence"

# These resource types can initiate or represent a request for care.  This
# research export must remain observational, even if future code adds entries.
FORBIDDEN_ORDER_RESOURCE_TYPES = frozenset(
    {
        "Appointment",
        "CarePlan",
        "CommunicationRequest",
        "DeviceRequest",
        "MedicationRequest",
        "NutritionOrder",
        "RequestGroup",
        "ServiceRequest",
        "SupplyRequest",
        "Task",
        "VisionPrescription",
    }
)


OMOPDomain = Literal["Condition", "Death", "Drug", "Measurement", "Observation", "Procedure", "Visit"]
OMOPTable = Literal[
    "condition_occurrence",
    "death",
    "drug_exposure",
    "measurement",
    "observation",
    "procedure_occurrence",
    "visit_occurrence",
]


class OMOPStyleRecord(StrictModel):
    """A compact internal analytical record inspired by OMOP CDM domains.

    This is intentionally not claimed to be a complete OMOP CDM row.  The
    ``patient_key_local`` field is an approved-environment join key and must
    not be copied into an exchange payload.
    """

    record_id: str
    patient_key_local: str = Field(min_length=3)
    domain: OMOPDomain
    table: OMOPTable
    event_type: EventType
    event_date: date | None = None
    event_datetime: datetime | None = None
    start_date: date | None = None
    start_datetime: datetime | None = None
    end_date: date | None = None
    end_datetime: datetime | None = None
    temporal_role: Literal["point", "start", "end"]
    concept_id: int | None = Field(default=None, ge=0)
    source_value: str
    value_as_number: float | None = None
    value_as_string: str | None = None
    unit_source_value: str | None = None
    confidence: float = Field(ge=0, le=1)
    time_precision: str


_OMOP_MAPPING: dict[EventType, tuple[OMOPDomain, OMOPTable]] = {
    EventType.DIAGNOSIS: ("Condition", "condition_occurrence"),
    EventType.METASTASIS: ("Condition", "condition_occurrence"),
    EventType.PROGRESSION: ("Condition", "condition_occurrence"),
    EventType.ECOG: ("Observation", "observation"),
    EventType.PATHOLOGY_STAGE: ("Observation", "observation"),
    EventType.MOLECULAR_TEST: ("Observation", "observation"),
    EventType.RESPONSE: ("Observation", "observation"),
    EventType.TOXICITY: ("Observation", "observation"),
    EventType.LABORATORY: ("Measurement", "measurement"),
    EventType.SURGERY: ("Procedure", "procedure_occurrence"),
    EventType.TREATMENT_START: ("Drug", "drug_exposure"),
    EventType.TREATMENT_END: ("Drug", "drug_exposure"),
    EventType.FOLLOW_UP: ("Visit", "visit_occurrence"),
    EventType.DEATH: ("Death", "death"),
}


_FHIR_MAPPING: dict[EventType, str] = {
    EventType.DIAGNOSIS: "Condition",
    EventType.METASTASIS: "Condition",
    EventType.PROGRESSION: "Condition",
    EventType.ECOG: "Observation",
    EventType.PATHOLOGY_STAGE: "Observation",
    EventType.MOLECULAR_TEST: "Observation",
    EventType.LABORATORY: "Observation",
    EventType.RESPONSE: "Observation",
    EventType.TOXICITY: "Observation",
    EventType.DEATH: "Observation",
    EventType.SURGERY: "Procedure",
    EventType.TREATMENT_START: "MedicationAdministration",
    EventType.TREATMENT_END: "MedicationAdministration",
    EventType.FOLLOW_UP: "Encounter",
}


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _source_value(event: ClinicalEvent) -> str:
    """Return a structured source value, never an EvidenceSpan excerpt."""

    if event.code:
        return event.code
    if event.value is not None:
        return str(event.value)
    return _enum_value(event.event_type)


def clinical_event_to_omop(
    event: ClinicalEvent,
    *,
    concept_id: int | None = None,
) -> OMOPStyleRecord:
    """Convert one event to an internal OMOP-style analytical record.

    No concept is inferred.  A reviewed vocabulary mapping may be supplied by
    the caller; otherwise ``concept_id`` remains explicitly null.
    """

    domain, table = _OMOP_MAPPING[event.event_type]
    event_datetime = event.event_time
    event_date = event_datetime.date() if event_datetime is not None else None
    temporal_role: Literal["point", "start", "end"] = "point"
    start_datetime: datetime | None = event_datetime
    end_datetime: datetime | None = None
    if event.event_type is EventType.TREATMENT_START:
        temporal_role = "start"
    elif event.event_type is EventType.TREATMENT_END:
        temporal_role = "end"
        start_datetime = None
        end_datetime = event_datetime

    value_as_number: float | None = None
    value_as_string: str | None = None
    if isinstance(event.value, bool):
        value_as_string = str(event.value).lower()
    elif isinstance(event.value, (int, float)):
        value_as_number = float(event.value)
    elif event.value is not None:
        value_as_string = str(event.value)

    return OMOPStyleRecord(
        record_id=event.id,
        patient_key_local=event.patient_key,
        domain=domain,
        table=table,
        event_type=event.event_type,
        event_date=event_date,
        event_datetime=event_datetime,
        start_date=start_datetime.date() if start_datetime else None,
        start_datetime=start_datetime,
        end_date=end_datetime.date() if end_datetime else None,
        end_datetime=end_datetime,
        temporal_role=temporal_role,
        concept_id=concept_id,
        source_value=_source_value(event),
        value_as_number=value_as_number,
        value_as_string=value_as_string,
        unit_source_value=event.unit,
        confidence=event.confidence,
        time_precision=_enum_value(event.time_precision),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fhir_id(prefix: str, material: str) -> str:
    # FHIR ids permit only letters, digits, hyphens, and periods and are at
    # most 64 characters.  Hashing also keeps local identifiers out of output.
    return f"{prefix}-{_sha256(material)[:40]}"


def _subject_reference(patient_key: str) -> dict[str, str]:
    return {"reference": f"Patient/{_fhir_id('subject', patient_key)}"}


def _fhir_datetime(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _nested_extension(url: str, value_key: str, value: Any) -> dict[str, Any]:
    return {"url": url, value_key: value}


_CELL_REFERENCE_RE = re.compile(r"^[A-Za-z]{1,4}[1-9][0-9]{0,7}$")


def _evidence_extension(span: EvidenceSpan) -> dict[str, Any]:
    """Serialize provenance locators without excerpt or source document text."""

    locators: list[dict[str, Any]] = [
        _nested_extension(
            "source-document-sha256",
            "valueString",
            _sha256(span.source_document_id),
        ),
        _nested_extension("start-offset", "valueInteger", span.start),
        _nested_extension("end-offset", "valueInteger", span.end),
        _nested_extension("text-sha256", "valueString", span.text_sha256.lower()),
    ]
    if span.sheet_name:
        # A worksheet name is a source locator, not clinical prose.  Hashing it
        # still avoids leaking an accidentally identifying worksheet title.
        locators.append(
            _nested_extension(
                "sheet-name-sha256", "valueString", _sha256(span.sheet_name)
            )
        )
    if span.cell_ref:
        if _CELL_REFERENCE_RE.fullmatch(span.cell_ref):
            locators.append(
                _nested_extension("cell-reference", "valueString", span.cell_ref.upper())
            )
        else:
            locators.append(
                _nested_extension(
                    "cell-reference-sha256", "valueString", _sha256(span.cell_ref)
                )
            )
    return {"url": FHIR_EVIDENCE_SPAN_URL, "extension": locators}


def _codeable_concept(event: ClinicalEvent) -> dict[str, Any]:
    code = event.code or _enum_value(event.event_type)
    concept: dict[str, Any] = {
        "coding": [
            {
                "system": FHIR_EVENT_CODE_SYSTEM,
                "code": code,
            }
        ]
    }
    # ``code`` is expected to be a structured source value in ClinicalEvent.
    # Do not populate narrative text from an evidence excerpt.
    if event.code:
        concept["text"] = event.code
    return concept


def _value_element(event: ClinicalEvent) -> dict[str, Any]:
    value = event.value
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"valueBoolean": value}
    if isinstance(value, (int, float)):
        if event.unit:
            return {"valueQuantity": {"value": value, "unit": event.unit}}
        if isinstance(value, int):
            return {"valueInteger": value}
        return {"valueQuantity": {"value": value}}
    return {"valueString": str(value)}


def _base_resource(event: ClinicalEvent, resource_type: str) -> dict[str, Any]:
    extensions = [_evidence_extension(span) for span in event.evidence_spans]
    extensions.extend(
        [
            {
                "url": FHIR_TIME_PRECISION_URL,
                "valueCode": _enum_value(event.time_precision),
            },
            {
                "url": FHIR_CONFIDENCE_URL,
                "valueDecimal": event.confidence,
            },
        ]
    )
    return {
        "resourceType": resource_type,
        "id": _fhir_id("event", event.id),
        "meta": {
            "tag": [
                {
                    "system": f"{FHIR_BASE_URL}/CodeSystem/intended-use",
                    "code": "research-only",
                }
            ]
        },
        "extension": extensions,
        "subject": _subject_reference(event.patient_key),
    }


def _condition(event: ClinicalEvent) -> dict[str, Any]:
    resource = _base_resource(event, "Condition")
    resource.update(
        {
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "confirmed",
                    }
                ]
            },
            "code": _codeable_concept(event),
        }
    )
    if event.event_time:
        resource["onsetDateTime"] = _fhir_datetime(event.event_time)
        resource["recordedDate"] = _fhir_datetime(event.event_time)
    return resource


def _observation(event: ClinicalEvent) -> dict[str, Any]:
    resource = _base_resource(event, "Observation")
    resource.update(
        {
            "status": "final",
            "code": _codeable_concept(event),
            **_value_element(event),
        }
    )
    if event.event_time:
        resource["effectiveDateTime"] = _fhir_datetime(event.event_time)
    return resource


def _procedure(event: ClinicalEvent) -> dict[str, Any]:
    resource = _base_resource(event, "Procedure")
    resource.update({"status": "completed", "code": _codeable_concept(event)})
    if event.event_time:
        resource["performedDateTime"] = _fhir_datetime(event.event_time)
    return resource


def _medication_administration(event: ClinicalEvent) -> dict[str, Any]:
    resource = _base_resource(event, "MedicationAdministration")
    resource.update(
        {
            "status": (
                "in-progress"
                if event.event_type is EventType.TREATMENT_START
                else "stopped"
            ),
            "medicationCodeableConcept": _codeable_concept(event),
        }
    )
    if event.event_time:
        resource["effectiveDateTime"] = _fhir_datetime(event.event_time)
    else:
        # MedicationAdministration.effective[x] is required in R4.  Represent
        # missing time with the standard data-absent-reason extension rather
        # than inventing a date.
        resource["effectiveDateTime"] = None
        resource["_effectiveDateTime"] = {
            "extension": [
                {
                    "url": "http://hl7.org/fhir/StructureDefinition/data-absent-reason",
                    "valueCode": "unknown",
                }
            ]
        }
    return resource


def _encounter(event: ClinicalEvent) -> dict[str, Any]:
    resource = _base_resource(event, "Encounter")
    resource.update(
        {
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
                "display": "ambulatory",
            },
            "type": [_codeable_concept(event)],
        }
    )
    if event.event_time:
        instant = _fhir_datetime(event.event_time)
        resource["period"] = {"start": instant, "end": instant}
    return resource


def clinical_event_to_fhir(event: ClinicalEvent) -> dict[str, Any]:
    """Convert a clinical event to a non-ordering FHIR R4 resource."""

    resource_type = _FHIR_MAPPING[event.event_type]
    converters = {
        "Condition": _condition,
        "Observation": _observation,
        "Procedure": _procedure,
        "MedicationAdministration": _medication_administration,
        "Encounter": _encounter,
    }
    resource = converters[resource_type](event)
    _assert_no_order_resources(resource)
    return resource


def _assert_no_order_resources(value: Any) -> None:
    """Reject a forbidden resource anywhere in a nested export."""

    if isinstance(value, dict):
        resource_type = value.get("resourceType")
        if resource_type in FORBIDDEN_ORDER_RESOURCE_TYPES:
            raise ValueError(f"order resource is forbidden: {resource_type}")
        for child in value.values():
            _assert_no_order_resources(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_order_resources(child)


def _contains_local_patient_key(value: Any, patient_key: str) -> bool:
    """Find semantic identifier leakage without substring false positives."""

    if isinstance(value, dict):
        return any(
            _contains_local_patient_key(child, patient_key)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_local_patient_key(child, patient_key) for child in value)
    if not isinstance(value, str):
        return False
    return value == patient_key or value.endswith(f"/{patient_key}")


def patient_timeline_to_fhir_bundle(timeline: PatientTimeline) -> dict[str, Any]:
    """Serialize a timeline as a deterministic, research-only FHIR R4 Bundle.

    The bundle contains no Patient resource or direct patient key.  Every
    subject reference uses a digest-based local pseudonym, and EvidenceSpan
    excerpts are never serialized.
    """

    resources = [clinical_event_to_fhir(event) for event in timeline.events]
    bundle_material = "|".join([timeline.patient_key, *(event.id for event in timeline.events)])
    entries = [
        {
            "fullUrl": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, resource['id'])}",
            "resource": resource,
        }
        for resource in resources
    ]
    bundle: dict[str, Any] = {
        "resourceType": "Bundle",
        "id": _fhir_id("timeline", bundle_material),
        "meta": {
            "tag": [
                {
                    "system": f"{FHIR_BASE_URL}/CodeSystem/intended-use",
                    "code": "research-only",
                }
            ]
        },
        "identifier": {
            "system": f"{FHIR_BASE_URL}/identifier/timeline",
            "value": _sha256(bundle_material),
        },
        "type": "collection",
        "entry": entries,
    }
    _assert_no_order_resources(bundle)

    # Defense in depth: the local patient key must not appear anywhere in the
    # exchange serialization, even in an identifier or extension.
    if _contains_local_patient_key(bundle, timeline.patient_key):
        raise ValueError("local patient key leaked into FHIR exchange payload")
    return bundle


# Concise aliases for callers that use event/timeline terminology.
event_to_omop = clinical_event_to_omop
timeline_to_fhir_bundle = patient_timeline_to_fhir_bundle
timeline_to_fhir_r4_bundle = patient_timeline_to_fhir_bundle


__all__ = [
    "FHIR_EVIDENCE_SPAN_URL",
    "FORBIDDEN_ORDER_RESOURCE_TYPES",
    "OMOPStyleRecord",
    "clinical_event_to_fhir",
    "clinical_event_to_omop",
    "event_to_omop",
    "patient_timeline_to_fhir_bundle",
    "timeline_to_fhir_bundle",
    "timeline_to_fhir_r4_bundle",
]
