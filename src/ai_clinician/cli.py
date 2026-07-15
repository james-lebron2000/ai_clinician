"""Local-only command line interface for patient-level clinical processing."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from .causal import (
    CausalAdmissionThresholds,
    CausalStrategyEngine,
)
from .data import link_patient_views, profile_to_dict, profile_workbooks
from .feature_manifest import AnalysisManifestRegistration, parse_analysis_manifest
from .guidelines import GuidelineRegistry, export_cpg_on_fhir
from .identity import GovernanceDirectory
from .models import (
    CandidateDiseaseState,
    CausalEffectEstimate,
    CohortSnapshot,
    DataProfileRegistration,
    DecisionPointType,
    ExpertReview,
    GuidelineChannel,
    GuidelineRecommendation,
    GuidelineRelease,
    ModelRelease,
    PatientTimeline,
    ReviewDecision,
    ReviewRole,
    ReleaseStatus,
    StateValidationReport,
    StatePartitionSnapshot,
    TargetTrialSpec,
    TargetTrialResult,
    TargetTrialInputSnapshot,
    TimelineValidationManifestRegistration,
)
from .privacy import PrivacyConfig, pseudonymize_identifier
from .states import (
    ContinuousTimeStateDiscovery,
    assert_no_future_information,
    chronological_patient_split,
    evaluate_state_validation_gate,
)
from .store import ResearchStore
from .timeline import build_timeline
from .validation import cohen_kappa, evaluate_gold_standard


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _emit(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True), file=stream)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _open_private_text(path: Path) -> Any:
    """Open a derived artifact with owner-only permissions, including overwrites."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        raise


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _open_private_text(path) as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _open_private_text(path) as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _privacy() -> PrivacyConfig:
    return PrivacyConfig.from_env()


def _as_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _data_profile(args: argparse.Namespace) -> int:
    privacy = _privacy()
    workbooks = [privacy.resolve_raw(path) for path in args.workbook]
    profile_options: dict[str, Any] = {"header_rows": args.header_rows}
    if args.id_column:
        profile_options["identifier_aliases"] = tuple(args.id_column)
    profile = profile_workbooks(workbooks, **profile_options)
    payload = profile_to_dict(profile)
    registration: DataProfileRegistration | None = None
    if args.output:
        _write_json(privacy.resolve_derived(args.output, create_parent=True), payload)
    if args.register:
        source_snapshot_sha256 = hashlib.sha256(
            json.dumps(
                sorted(workbook.source_document_id for workbook in profile.workbooks),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        registration = DataProfileRegistration(
            source_snapshot_sha256=source_snapshot_sha256,
            workbook_count=len(profile.workbooks),
            sheet_count=sum(
                len(workbook.sheet_profiles) for workbook in profile.workbooks
            ),
            aggregate_patient_count=profile.union_patient_count,
            quarantined_row_count=sum(
                workbook.quarantine.total_rows for workbook in profile.workbooks
            ),
        )
        identity = GovernanceDirectory.from_env().authenticate(
            os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
            allowed_roles=["data_steward"],
        )
        _store(args).put(
            "data_profile",
            registration.id,
            registration,
            actor=identity.subject,
            governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        )
    _emit(
        {
            "status": "profiled",
            "workbook_count": len(profile.workbooks),
            "aggregate_distinct_patient_count": profile.union_patient_count,
            "quarantined_row_count": sum(
                workbook.quarantine.total_rows for workbook in profile.workbooks
            ),
            "output_written": bool(args.output),
            "data_profile_id": registration.id if registration else None,
            "source_snapshot_sha256": (
                registration.source_snapshot_sha256 if registration else None
            ),
        }
    )
    return 0


def _data_link(args: argparse.Namespace) -> int:
    privacy = _privacy()
    workbooks = [privacy.resolve_raw(path) for path in args.workbook]
    secret = os.environ.get("AI_CLINICIAN_PSEUDONYM_SECRET", "")
    field_aliases: dict[str, str] = {}
    if args.field_aliases:
        aliases = _read_json(privacy.resolve_derived(args.field_aliases))
        if not isinstance(aliases, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in aliases.items()
        ):
            raise ValueError("field aliases must be a string-to-string JSON object")
        field_aliases = aliases
    linkage = link_patient_views(
        workbooks,
        hmac_secret=secret,
        identifier_header=args.id_column,
        header_rows=args.header_rows,
        field_aliases=field_aliases,
    )
    _write_json(
        privacy.resolve_derived(args.output, create_parent=True), linkage
    )
    _emit(
        {
            "status": "linked",
            "patient_view_count": len(linkage.patient_views),
            "quarantined_row_count": len(linkage.quarantine),
            "conflict_count": sum(
                len(view.conflicts) for view in linkage.patient_views
            ),
            "source_document_count": linkage.source_document_count,
            "output_written": True,
        }
    )
    return 0


def _data_lock_cohort(args: argparse.Namespace) -> int:
    privacy = _privacy()
    document = _read_json(privacy.resolve_derived(args.input))
    if not isinstance(document, dict):
        raise ValueError("cohort lock input must be an object")
    membership_path = privacy.resolve_derived(str(document.get("membership_file", "")))
    membership = _read_json(membership_path)
    normalized_membership = [str(item).strip() for item in membership] if isinstance(membership, list) else []
    if (
        not isinstance(membership, list)
        or not membership
        or any(not item for item in normalized_membership)
        or len(normalized_membership) != len(set(normalized_membership))
    ):
        raise ValueError("cohort membership must be a non-empty unique patient-key list")
    membership_hash = hashlib.sha256(
        json.dumps(
            sorted(normalized_membership),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    timeline_path = privacy.resolve_derived(str(document.get("timeline_snapshot_file", "")))
    split_path = privacy.resolve_derived(str(document.get("split_manifest_file", "")))
    split_manifest = _read_json(split_path)
    if not isinstance(split_manifest, dict) or not isinstance(
        split_manifest.get("index_dates"), dict
    ):
        raise ValueError("split manifest requires patient lists and first-decision index_dates")
    split_names = ("development", "tuning", "locked_test")
    split_members: dict[str, list[str]] = {}
    for split_name in split_names:
        raw_members = split_manifest.get(split_name)
        if not isinstance(raw_members, list):
            raise ValueError("split manifest requires all three patient-level partitions")
        split_members[split_name] = [str(item).strip() for item in raw_members]
        if any(not item for item in split_members[split_name]) or len(
            split_members[split_name]
        ) != len(set(split_members[split_name])):
            raise ValueError("split manifest partitions require unique patient keys")
    partition_sets = [set(split_members[name]) for name in split_names]
    if (
        partition_sets[0] & partition_sets[1]
        or partition_sets[0] & partition_sets[2]
        or partition_sets[1] & partition_sets[2]
        or set().union(*partition_sets) != set(normalized_membership)
    ):
        raise ValueError("split manifest must partition the locked cohort exactly once")
    index_dates = {
        str(key).strip(): value
        for key, value in split_manifest["index_dates"].items()
    }
    if set(index_dates) != set(normalized_membership):
        raise ValueError("split index dates must cover every locked patient exactly once")
    expected_partition = chronological_patient_split(
        [
            {"patient_id": patient_id, "decision_date": index_dates[patient_id]}
            for patient_id in normalized_membership
        ]
    )
    if (
        tuple(split_members["development"]) != expected_partition.development
        or tuple(split_members["tuning"]) != expected_partition.tuning
        or tuple(split_members["locked_test"]) != expected_partition.locked_test
    ):
        raise ValueError(
            "split manifest must be the deterministic 70/15/15 chronological split"
        )
    membership_digest = lambda values: hashlib.sha256(
        json.dumps(
            sorted(values), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    cohort_fields: dict[str, Any] = {
        "study_id": document["study_id"],
        "data_profile_id": document["data_profile_id"],
        "source_data_sha256": document["source_data_sha256"],
        "patient_membership_sha256": membership_hash,
        "timeline_snapshot_sha256": hashlib.sha256(
            timeline_path.read_bytes()
        ).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "development_membership_sha256": membership_digest(
            split_members["development"]
        ),
        "tuning_membership_sha256": membership_digest(split_members["tuning"]),
        "locked_test_membership_sha256": membership_digest(
            split_members["locked_test"]
        ),
        "development_patient_count": len(split_members["development"]),
        "tuning_patient_count": len(split_members["tuning"]),
        "locked_test_patient_count": len(split_members["locked_test"]),
        "eligible_patient_count": len(normalized_membership),
        "locked_at": datetime.now().astimezone(),
    }
    if document.get("id"):
        cohort_fields["id"] = document["id"]
    cohort = CohortSnapshot(**cohort_fields)
    identity = GovernanceDirectory.from_env().authenticate(
        os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        allowed_roles=["data_steward"],
    )
    stored = _store(args).put(
        "cohort_snapshot",
        cohort.id,
        cohort,
        actor=identity.subject,
        governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
    )
    _emit(
        {
            **stored,
            "cohort_snapshot_hash": cohort.content_hash(),
            "eligible_patient_count": cohort.eligible_patient_count,
            "research_only": True,
        }
    )
    return 0


def _timeline_build(args: argparse.Namespace) -> int:
    privacy = _privacy()
    input_path = privacy.resolve_raw(args.input)
    output_path = privacy.resolve_derived(args.output, create_parent=True)
    secret = os.environ.get("AI_CLINICIAN_PSEUDONYM_SECRET", "")
    timelines: list[PatientTimeline] = []
    event_count = 0
    for record in _read_jsonl(input_path):
        patient_identifier = str(record.get("patient_id", ""))
        patient_key = pseudonymize_identifier(patient_identifier, secret=secret)
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("each timeline input row requires text")
        timeline = build_timeline(
            patient_key,
            text,
            source_document_id=str(record.get("source_document_id", "")),
            recorded_at=_as_datetime(record.get("recorded_at")),
            sheet_name=record.get("sheet_name"),
            cell_ref=record.get("cell_ref"),
        )
        timelines.append(timeline)
        event_count += len(timeline.events)
    _write_jsonl(output_path, timelines)
    _emit(
        {
            "status": "built",
            "timeline_count": len(timelines),
            "event_count": event_count,
            "output_written": True,
        }
    )
    return 0


def _timeline_manifest_registration(
    *,
    gold: list[PatientTimeline],
    gold_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    registration_id: str | None = None,
) -> tuple[TimelineValidationManifestRegistration, set[str]]:
    locked_keys = manifest.get("locked_patient_keys")
    if not isinstance(locked_keys, list) or len(locked_keys) != len(set(locked_keys)):
        raise ValueError("decision manifest requires unique locked_patient_keys")
    gold_keys = {timeline.patient_key for timeline in gold}
    locked_key_set = {str(item) for item in locked_keys}
    if not locked_key_set.issubset(gold_keys):
        raise ValueError("locked split contains a patient outside the gold snapshot")
    if len(locked_key_set) * 5 != len(gold):
        raise ValueError("locked split must be exactly 20% of the 500-case gold set")
    annotation_pairs = manifest.get("annotation_pairs")
    if not isinstance(annotation_pairs, list) or len(annotation_pairs) < 500:
        raise ValueError("decision manifest requires at least 500 paired annotations")
    first_labels: list[str] = []
    second_labels: list[str] = []
    annotation_case_keys: set[str] = set()
    annotation_reviewers: set[str] = set()
    for pair in annotation_pairs:
        if not isinstance(pair, dict):
            raise ValueError("annotation pairs must be structured records")
        first = str(pair.get("annotator_a_label", "")).strip()
        second = str(pair.get("annotator_b_label", "")).strip()
        case_key = str(pair.get("case_key", "")).strip()
        reviewer_a = str(pair.get("reviewer_a_key", "")).strip()
        reviewer_b = str(pair.get("reviewer_b_key", "")).strip()
        if (
            not first
            or not second
            or not case_key
            or not reviewer_a
            or not reviewer_b
            or reviewer_a == reviewer_b
            or case_key in annotation_case_keys
        ):
            raise ValueError(
                "annotation pairs require one unique case and two independent reviewers"
            )
        annotation_case_keys.add(case_key)
        annotation_reviewers.update((reviewer_a, reviewer_b))
        if first != second:
            arbitrator = str(pair.get("arbitrator_key", "")).strip()
            adjudicated = str(pair.get("adjudicated_label", "")).strip()
            if (
                not arbitrator
                or arbitrator in {reviewer_a, reviewer_b}
                or not adjudicated
            ):
                raise ValueError(
                    "disagreements require a distinct third reviewer and adjudication"
                )
            annotation_reviewers.add(arbitrator)
        first_labels.append(first)
        second_labels.append(second)
    if annotation_case_keys != gold_keys:
        raise ValueError("paired annotations must cover every gold case exactly once")
    if len(annotation_reviewers) < 3:
        raise ValueError("gold annotation requires at least three CRC experts")
    if len(set(first_labels) | set(second_labels)) < 2:
        raise ValueError("annotation kappa requires at least two observed labels")
    locked_split_sha256 = hashlib.sha256(
        json.dumps(
            sorted(locked_key_set),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    fields: dict[str, Any] = {
        "gold_snapshot_id": str(manifest.get("gold_snapshot_id", "")),
        "gold_snapshot_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "decision_view_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "locked_split_sha256": locked_split_sha256,
        "gold_total_case_count": len(gold),
        "locked_case_count": len(locked_key_set),
        "annotation_pair_count": len(annotation_pairs),
        "annotation_kappa": cohen_kappa(first_labels, second_labels),
        "extractor_release_id": str(manifest.get("extractor_release_id", "")),
        "extractor_sha256": str(manifest.get("extractor_sha256", "")),
        "evaluation_code_sha256": hashlib.sha256(
            Path(__file__).with_name("validation.py").read_bytes()
        ).hexdigest(),
    }
    if registration_id:
        fields["id"] = registration_id
    return TimelineValidationManifestRegistration(**fields), locked_key_set


def _timeline_register_manifest(args: argparse.Namespace) -> int:
    privacy = _privacy()
    gold_path = privacy.resolve_derived(args.gold)
    manifest_path = privacy.resolve_derived(args.decision_manifest)
    gold = [PatientTimeline.model_validate(row) for row in _read_jsonl(gold_path)]
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("decision_views"), list
    ):
        raise ValueError("decision manifest must contain decision_views")
    registration, _ = _timeline_manifest_registration(
        gold=gold,
        gold_path=gold_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    identity = GovernanceDirectory.from_env().authenticate(
        os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        allowed_roles=["data_steward"],
    )
    stored = _store(args).put(
        "timeline_validation_manifest",
        registration.id,
        registration,
        actor=identity.subject,
        governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
    )
    _emit(
        {
            **stored,
            "annotation_kappa": registration.annotation_kappa,
            "gold_total_case_count": registration.gold_total_case_count,
            "locked_case_count": registration.locked_case_count,
            "research_only": True,
        }
    )
    return 0


def _timeline_validate(args: argparse.Namespace) -> int:
    privacy = _privacy()
    predicted_path = privacy.resolve_derived(args.predicted)
    gold_path = privacy.resolve_derived(args.gold)
    manifest_path = privacy.resolve_derived(args.decision_manifest)
    predicted = [
        PatientTimeline.model_validate(row)
        for row in _read_jsonl(predicted_path)
    ]
    gold = [
        PatientTimeline.model_validate(row)
        for row in _read_jsonl(gold_path)
    ]
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("decision_views"), list
    ):
        raise ValueError("decision manifest must contain decision_views")
    registration = TimelineValidationManifestRegistration.model_validate(
        _require_signed_record(
            _store(args),
            "timeline_validation_manifest",
            args.manifest_registration_id,
        )["payload"]
    )
    recomputed_registration, locked_key_set = _timeline_manifest_registration(
        gold=gold,
        gold_path=gold_path,
        manifest=manifest,
        manifest_path=manifest_path,
        registration_id=registration.id,
    )
    if registration.content_hash() != recomputed_registration.content_hash():
        raise ValueError("timeline validation manifest differs from its locked registration")
    expected_review_hash = registration.content_hash()
    annotation_reviews: list[ExpertReview] = []
    store = _store(args)
    for record in store.list_all_latest("expert_review"):
        payload = record["payload"]
        if payload.get("object_id") != registration.id:
            continue
        if not store.artifact_has_valid_signature("expert_review", record["object_id"]):
            raise ValueError("annotation panel review is not signature-bound")
        review = ExpertReview.model_validate(payload)
        if (
            review.object_hash != expected_review_hash
            or review.role is not ReviewRole.CRC_CLINICAL_EXPERT
            or review.decision is not ReviewDecision.APPROVE
            or not review.independent
        ):
            raise ValueError("annotation panel review is invalid")
        annotation_reviews.append(review)
    if len({review.reviewer_key for review in annotation_reviews}) < 3:
        raise ValueError("timeline validation requires three independent CRC reviews")
    locked_gold = [
        timeline for timeline in gold if timeline.patient_key in locked_key_set
    ]
    locked_predicted = [
        timeline for timeline in predicted if timeline.patient_key in locked_key_set
    ]
    locked_decision_views = [
        view
        for view in manifest["decision_views"]
        if str(view.get("patient_key", "")) in locked_key_set
    ]
    report = evaluate_gold_standard(
        locked_predicted,
        locked_gold,
        decision_views=locked_decision_views,
        require_decision_views=True,
        gold_total_case_count=len(gold),
        annotation_kappa=registration.annotation_kappa,
        gold_snapshot_id=registration.gold_snapshot_id,
        gold_snapshot_sha256=registration.gold_snapshot_sha256,
        locked_split_sha256=registration.locked_split_sha256,
        locked_fraction=0.20,
        decision_view_manifest_sha256=registration.decision_view_manifest_sha256,
        extractor_release_id=registration.extractor_release_id,
        extractor_sha256=registration.extractor_sha256,
        evaluation_code_sha256=registration.evaluation_code_sha256,
        manifest_registration_id=registration.id,
        manifest_registration_hash=registration.content_hash(),
        annotation_review_ids=[review.id for review in annotation_reviews],
    )
    if args.output:
        _write_json(privacy.resolve_derived(args.output, create_parent=True), report)
    if args.register:
        identity = GovernanceDirectory.from_env().authenticate(
            os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
            allowed_roles=["data_steward", "methodologist"],
        )
        _store(args).put(
            "timeline_quality_report",
            report.id,
            report,
            actor=identity.subject,
            governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        )
    _emit(report)
    return 0 if report.passed else 2


def _states_register_manifest(args: argparse.Namespace) -> int:
    privacy = _privacy()
    document = _read_json(privacy.resolve_derived(args.input))
    if not isinstance(document, dict):
        raise ValueError("state manifest registration input must be an object")
    store = _store(args)
    cohort = CohortSnapshot.model_validate(
        _require_signed_record(
            store, "cohort_snapshot", str(document.get("cohort_snapshot_id", ""))
        )["payload"]
    )
    schema_path = privacy.resolve_derived(str(document.get("source_schema_file", "")))
    registration = AnalysisManifestRegistration(
        manifest=parse_analysis_manifest(document.get("analysis_column_manifest")),
        cohort_snapshot_id=cohort.id,
        cohort_snapshot_hash=cohort.content_hash(),
        source_schema_sha256=hashlib.sha256(schema_path.read_bytes()).hexdigest(),
    )
    identity = GovernanceDirectory.from_env().authenticate(
        os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        allowed_roles=["data_steward"],
    )
    stored = store.put(
        "analysis_manifest_registration",
        registration.id,
        registration,
        actor=identity.subject,
        governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
    )
    _emit(
        {
            **stored,
            "analysis_manifest_registration_hash": registration.content_hash(),
            "analysis_column_manifest_sha256": registration.manifest.sha256,
            "research_only": True,
        }
    )
    return 0


def _states_fit(args: argparse.Namespace) -> int:
    privacy = _privacy()
    document = _read_json(privacy.resolve_derived(args.input))
    if any(
        key in document
        for key in ("analysis_column_manifest", "column_manifest", "feature_manifest")
    ):
        raise ValueError("state fit must load the signed manifest registration")
    store = _store(args)
    registration = AnalysisManifestRegistration.model_validate(
        _require_signed_record(
            store,
            "analysis_manifest_registration",
            str(document.get("analysis_manifest_registration_id", "")),
        )["payload"]
    )
    if document.get("analysis_manifest_registration_hash") != registration.content_hash():
        raise ValueError("analysis manifest registration hash mismatch")
    expected_review_hash = registration.content_hash()
    manifest_reviews = [
        ExpertReview.model_validate(record["payload"])
        for record in store.list_all_latest("expert_review")
        if record["payload"].get("object_id") == registration.id
        and store.artifact_has_valid_signature("expert_review", record["object_id"])
    ]
    if not any(
        review.object_hash == expected_review_hash
        and review.role is ReviewRole.CRC_CLINICAL_EXPERT
        and review.decision is ReviewDecision.APPROVE
        and review.independent
        for review in manifest_reviews
    ):
        raise ValueError("state column roles require a signed CRC expert review")
    cohort = CohortSnapshot.model_validate(
        _require_signed_record(store, "cohort_snapshot", registration.cohort_snapshot_id)[
            "payload"
        ]
    )
    if cohort.content_hash() != registration.cohort_snapshot_hash:
        raise ValueError("state manifest cohort snapshot hash mismatch")
    schema_path = privacy.resolve_derived(str(document.get("source_schema_file", "")))
    if hashlib.sha256(schema_path.read_bytes()).hexdigest() != registration.source_schema_sha256:
        raise ValueError("source schema differs from manifest registration")
    manifest = registration.manifest
    manifest_hash = manifest.sha256
    records = document["records"]
    patient_id_column = document.get("patient_id_column", "patient_id")
    time_column = document.get("time_column", "event_time")
    index_date_column = document.get("index_date_column", "decision_date")
    availability_column = document.get(
        "feature_available_time_column", "feature_available_time"
    )
    assert_no_future_information(
        records,
        event_time_column=time_column,
        feature_available_time_column=availability_column,
    )
    partition = chronological_patient_split(
        records,
        patient_id_column=patient_id_column,
        index_date_column=index_date_column,
    )
    development_ids = set(partition.development)
    tuning_ids = set(partition.tuning)
    locked_ids = set(partition.locked_test)
    def membership_digest(values: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                sorted(values), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    membership = _read_json(
        privacy.resolve_derived(str(document.get("cohort_membership_file", "")))
    )
    membership_set = {str(item) for item in membership} if isinstance(membership, list) else set()
    membership_hash = hashlib.sha256(
        json.dumps(
            sorted(membership_set),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        membership_hash != cohort.patient_membership_sha256
        or set(partition.all_patient_ids) != membership_set
        or membership_digest(development_ids)
        != cohort.development_membership_sha256
        or membership_digest(tuning_ids) != cohort.tuning_membership_sha256
        or membership_digest(locked_ids) != cohort.locked_test_membership_sha256
        or len(development_ids) != cohort.development_patient_count
        or len(tuning_ids) != cohort.tuning_patient_count
        or len(locked_ids) != cohort.locked_test_patient_count
    ):
        raise ValueError("state analysis partition differs from the locked cohort")
    development_records = [
        row for row in records if str(row[patient_id_column]) in development_ids
    ]
    tuning_records = [
        row for row in records if str(row[patient_id_column]) in tuning_ids
    ]
    locked_records = [
        row for row in records if str(row[patient_id_column]) in locked_ids
    ]
    engine = ContinuousTimeStateDiscovery(
        n_states=int(document.get("n_states", 3)),
        random_seed=int(document.get("random_seed", 17)),
        bootstrap_replicates=int(document.get("bootstrap_replicates", 20)),
    )
    report = engine.fit(
        development_records,
        feature_names=document["feature_names"],
        patient_id_column=patient_id_column,
        time_column=time_column,
        index_date_column=index_date_column,
        feature_available_time_column=availability_column,
        treatment_columns=document["treatment_columns"],
        column_manifest=manifest,
        manifest_sha256=manifest_hash,
    )
    partition_snapshot = StatePartitionSnapshot(
        cohort_snapshot_id=cohort.id,
        cohort_snapshot_hash=cohort.content_hash(),
        analysis_manifest_registration_id=registration.id,
        analysis_manifest_registration_hash=registration.content_hash(),
        analytic_rows_sha256=hashlib.sha256(
            json.dumps(
                _jsonable(records),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        development_membership_sha256=membership_digest(development_ids),
        tuning_membership_sha256=membership_digest(tuning_ids),
        locked_test_membership_sha256=membership_digest(locked_ids),
        development_patient_count=len(development_ids),
        tuning_patient_count=len(tuning_ids),
        locked_test_patient_count=len(locked_ids),
        code_sha256=hashlib.sha256(
            Path(__file__).with_name("states.py").read_bytes()
        ).hexdigest(),
        locked_at=datetime.now().astimezone(),
    )
    identity = GovernanceDirectory.from_env().authenticate(
        os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        allowed_roles=["methodologist"],
    )
    store.put(
        "state_partition_snapshot",
        partition_snapshot.id,
        partition_snapshot,
        actor=identity.subject,
        governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
    )
    result = {
        "feature_names": report.model.feature_names,
        "treatment_columns": report.model.treatment_columns,
        "analysis_column_manifest": manifest.canonical_payload(),
        "analysis_column_manifest_sha256": report.analysis_manifest_sha256,
        "analysis_manifest_registration_id": registration.id,
        "analysis_manifest_registration_hash": registration.content_hash(),
        "state_partition_snapshot_id": partition_snapshot.id,
        "state_partition_snapshot_hash": partition_snapshot.content_hash(),
        # Backward-compatible name; the digest now binds every analysis
        # column and is always recomputed from canonical manifest content.
        "treatment_manifest_sha256": report.analysis_manifest_sha256,
        "state_labels": report.model.state_labels,
        "centroids": report.model.centroids,
        "transition_rates": report.model.transition_rates,
        "transition_counts": report.model.transition_counts,
        "person_time": report.model.person_time,
        "candidate_states": report.model.candidate_states,
        "state_transitions": report.model.state_transitions,
        "bootstrap_ari": report.bootstrap_ari,
        "labels": report.labels,
        "posterior_probabilities": report.posterior_probabilities,
        "tuning_labels": report.model.predict(tuning_records) if tuning_records else [],
        "partition_patient_counts": {
            "development": len(development_ids),
            "tuning": len(tuning_ids),
            "locked_test": len(locked_ids),
        },
        "research_only": True,
    }
    _write_json(privacy.resolve_derived(args.output, create_parent=True), result)
    _emit(
        {
            "status": "fitted",
            "development_observation_count": len(development_records),
            "tuning_observation_count": len(tuning_records),
            "locked_test_observation_count": len(locked_records),
            "state_count": len(report.model.state_labels),
            "bootstrap_ari": report.bootstrap_ari,
            "output_written": True,
        }
    )
    return 0


def _states_evaluate(args: argparse.Namespace) -> int:
    raise RuntimeError(
        "state promotion is disabled until the locked-test evaluator computes "
        "calibration, Brier gain, and review-bound interpretability evidence"
    )


def _target_trial_register_spec(args: argparse.Namespace) -> int:
    privacy = _privacy()
    document = _read_json(privacy.resolve_derived(args.input))
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("spec"), dict)
        or not isinstance(document.get("analysis_config"), dict)
    ):
        raise ValueError("protocol registration requires spec and analysis_config")
    store = _store(args)
    spec_fields = dict(document["spec"])
    cohort = CohortSnapshot.model_validate(
        _require_signed_record(
            store, "cohort_snapshot", str(spec_fields.get("cohort_snapshot_id", ""))
        )["payload"]
    )
    if spec_fields.get("study_id") != cohort.study_id:
        raise ValueError("target trial protocol belongs to another study")
    spec_fields["cohort_snapshot_hash"] = cohort.content_hash()
    spec_fields["data_snapshot_id"] = cohort.id
    spec_fields["analysis_code_sha256"] = hashlib.sha256(
        Path(__file__).with_name("causal.py").read_bytes()
    ).hexdigest()
    spec_fields["analysis_config_sha256"] = hashlib.sha256(
        json.dumps(
            document["analysis_config"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    baseline_release_record = _require_signed_record(
        store,
        "guideline_release",
        str(spec_fields.get("baseline_release_id", "")),
    )
    baseline_release = GuidelineRelease.model_validate(
        baseline_release_record["payload"]
    )
    if (
        baseline_release.channel is not GuidelineChannel.BASELINE_APPROVED
        or baseline_release.status not in {ReleaseStatus.LOCKED, ReleaseStatus.RELEASED}
    ):
        raise ValueError("target trial strategies require a locked approved baseline")
    baseline_ids = [
        str(item) for item in spec_fields.get("baseline_recommendation_ids", [])
    ]
    if not baseline_ids or not set(baseline_ids).issubset(
        baseline_release.recommendation_ids
    ):
        raise ValueError("target trial baseline recommendations are not in the release")
    baseline_hashes: dict[str, str] = {}
    supported_strategies: set[str] = set()
    decision_point = DecisionPointType(spec_fields.get("decision_point", ""))
    for recommendation_id in baseline_ids:
        recommendation = GuidelineRecommendation.model_validate(
            _require_signed_record(
                store, "guideline_recommendation", recommendation_id
            )["payload"]
        )
        if recommendation.channel is not GuidelineChannel.BASELINE_APPROVED:
            raise ValueError("target trial baseline recommendation has the wrong channel")
        recommendation_hash = recommendation.content_hash()
        if baseline_release.recommendation_hashes.get(
            recommendation_id
        ) != recommendation_hash:
            raise ValueError("baseline recommendation differs from the locked release")
        baseline_hashes[recommendation_id] = recommendation_hash
        supported_strategies.update(
            option.regimen_class
            for option in recommendation.treatment_options
            if option.guideline_backed and option.decision_point is decision_point
        )
    if not set(str(item) for item in spec_fields.get("strategies", [])).issubset(
        supported_strategies
    ):
        raise ValueError("target trial contains a strategy unsupported by the baseline")
    spec_fields["baseline_release_hash"] = baseline_release_record["content_sha256"]
    spec_fields["baseline_release_version"] = baseline_release_record["version"]
    spec_fields["baseline_recommendation_hashes"] = baseline_hashes
    spec = TargetTrialSpec.model_validate(spec_fields)
    if not spec.preregistered:
        raise ValueError("target trial protocol must be marked preregistered")
    identity = GovernanceDirectory.from_env().authenticate(
        os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        allowed_roles=["methodologist"],
    )
    stored = store.put(
        "target_trial_spec",
        spec.id,
        spec,
        actor=identity.subject,
        governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
    )
    _emit(
        {
            **stored,
            "target_trial_spec_hash": spec.content_hash(),
            "research_only": True,
        }
    )
    return 0


def _target_trial_freeze_input(args: argparse.Namespace) -> int:
    privacy = _privacy()
    document = _read_json(privacy.resolve_derived(args.input))
    if not isinstance(document, dict) or not isinstance(
        document.get("analysis_config"), dict
    ):
        raise ValueError("input freeze requires a preregistered analysis_config")
    store = _store(args)
    spec = TargetTrialSpec.model_validate(
        _require_signed_record(
            store, "target_trial_spec", str(document.get("target_trial_spec_id", ""))
        )["payload"]
    )
    analysis_config = document["analysis_config"]
    config_hash = hashlib.sha256(
        json.dumps(
            analysis_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    current_code_hash = hashlib.sha256(
        Path(__file__).with_name("causal.py").read_bytes()
    ).hexdigest()
    if (
        config_hash != spec.analysis_config_sha256
        or current_code_hash != spec.analysis_code_sha256
    ):
        raise ValueError("analysis code or configuration differs from preregistration")
    cohort = CohortSnapshot.model_validate(
        _require_signed_record(store, "cohort_snapshot", spec.cohort_snapshot_id)[
            "payload"
        ]
    )
    if cohort.content_hash() != spec.cohort_snapshot_hash:
        raise ValueError("cohort snapshot differs from preregistration")
    membership = _read_json(
        privacy.resolve_derived(str(document.get("cohort_membership_file", "")))
    )
    if not isinstance(membership, list) or len(membership) != len(
        set(str(item) for item in membership)
    ):
        raise ValueError("input freeze requires the locked cohort membership file")
    membership_set = {str(item) for item in membership}
    cohort_membership_hash = hashlib.sha256(
        json.dumps(
            sorted(membership_set),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if cohort_membership_hash != cohort.patient_membership_sha256:
        raise ValueError("cohort membership file differs from the locked snapshot")
    rows = _read_json(privacy.resolve_derived(str(document.get("rows_file", ""))))
    if not isinstance(rows, list):
        raise ValueError("analytic rows file must contain a list")
    patient_id_column = str(analysis_config.get("patient_id_column", "patient_id"))
    row_patient_ids = {str(row.get(patient_id_column, "")) for row in rows}
    if "" in row_patient_ids or not row_patient_ids.issubset(membership_set):
        raise ValueError("analytic rows include a patient outside the locked cohort")
    snapshot = TargetTrialInputSnapshot(
        target_trial_spec_id=spec.id,
        target_trial_spec_hash=spec.content_hash(),
        cohort_snapshot_id=cohort.id,
        cohort_snapshot_hash=cohort.content_hash(),
        analytic_rows_sha256=hashlib.sha256(
            json.dumps(
                _jsonable(rows),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        patient_membership_sha256=hashlib.sha256(
            json.dumps(
                sorted(row_patient_ids),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        eligible_patient_count=len(row_patient_ids),
    )
    identity = GovernanceDirectory.from_env().authenticate(
        os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        allowed_roles=["data_steward", "methodologist"],
    )
    stored = store.put(
        "target_trial_input_snapshot",
        snapshot.id,
        snapshot,
        actor=identity.subject,
        governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
    )
    _emit(
        {
            **stored,
            "target_trial_input_snapshot_hash": snapshot.content_hash(),
            "eligible_patient_count": snapshot.eligible_patient_count,
            "research_only": True,
        }
    )
    return 0


def _target_trial_run(args: argparse.Namespace) -> int:
    privacy = _privacy()
    document = _read_json(privacy.resolve_derived(args.input))
    if "spec" in document or "negative_control_passed" in document or "sensitivity_analysis_passed" in document:
        raise ValueError("protocol and audit pass flags must come from signed artifacts")
    store = _store(args)
    spec = TargetTrialSpec.model_validate(
        _require_signed_record(
            store, "target_trial_spec", str(document.get("target_trial_spec_id", ""))
        )["payload"]
    )
    if document.get("target_trial_spec_hash") != spec.content_hash():
        raise ValueError("target-trial specification hash mismatch")
    analysis_config = document.get("analysis_config")
    if not isinstance(analysis_config, dict):
        raise ValueError("run requires the preregistered analysis_config")
    analysis_config_hash = hashlib.sha256(
        json.dumps(
            analysis_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if analysis_config_hash != spec.analysis_config_sha256:
        raise ValueError("analysis configuration differs from preregistration")
    if hashlib.sha256(Path(__file__).with_name("causal.py").read_bytes()).hexdigest() != spec.analysis_code_sha256:
        raise ValueError("analysis code differs from preregistration")
    registered_input_snapshot = TargetTrialInputSnapshot.model_validate(
        _require_signed_record(
            store,
            "target_trial_input_snapshot",
            str(document.get("input_snapshot_id", "")),
        )["payload"]
    )
    if document.get("input_snapshot_hash") != registered_input_snapshot.content_hash():
        raise ValueError("target-trial input snapshot hash mismatch")
    cohort = CohortSnapshot.model_validate(
        _require_signed_record(store, "cohort_snapshot", spec.cohort_snapshot_id)[
            "payload"
        ]
    )
    if cohort.content_hash() != spec.cohort_snapshot_hash:
        raise ValueError("cohort snapshot differs from preregistration")
    membership = _read_json(
        privacy.resolve_derived(str(document.get("cohort_membership_file", "")))
    )
    if not isinstance(membership, list) or len(membership) != len(
        set(str(item) for item in membership)
    ):
        raise ValueError("run requires the locked cohort membership file")
    membership_set = {str(item) for item in membership}
    membership_hash = hashlib.sha256(
        json.dumps(
            sorted(membership_set),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if membership_hash != cohort.patient_membership_sha256:
        raise ValueError("cohort membership file differs from the locked snapshot")
    rows = _read_json(
        privacy.resolve_derived(str(document.get("rows_file", "")))
    )
    if not isinstance(rows, list):
        raise ValueError("target-trial input rows are required")
    patient_id_column = str(analysis_config.get("patient_id_column", "patient_id"))
    row_patient_ids = {str(row.get(patient_id_column, "")) for row in rows}
    if "" in row_patient_ids or not row_patient_ids.issubset(membership_set):
        raise ValueError("analytic rows include a patient outside the locked cohort")
    input_snapshot = TargetTrialInputSnapshot(
        id=registered_input_snapshot.id,
        target_trial_spec_id=spec.id,
        target_trial_spec_hash=spec.content_hash(),
        cohort_snapshot_id=cohort.id,
        cohort_snapshot_hash=cohort.content_hash(),
        analytic_rows_sha256=hashlib.sha256(
            json.dumps(
                _jsonable(rows),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        patient_membership_sha256=hashlib.sha256(
            json.dumps(
                sorted(row_patient_ids),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        eligible_patient_count=len(row_patient_ids),
    )
    if input_snapshot.content_hash() != registered_input_snapshot.content_hash():
        raise ValueError("analytic rows differ from the frozen input snapshot")
    thresholds = CausalAdmissionThresholds(**analysis_config.get("thresholds", {}))
    result = CausalStrategyEngine(
        thresholds=thresholds,
        cross_fit_folds=int(analysis_config.get("cross_fit_folds", 5)),
        random_seed=int(analysis_config.get("random_seed", 29)),
    ).run(
        spec,
        rows,
        action_column=analysis_config.get("action_column", "treatment_id"),
        outcome_column=analysis_config.get("outcome_column"),
        patient_id_column=patient_id_column,
        ood_column=analysis_config.get("ood_column", "is_ood"),
        confounders=spec.confounders,
        metric_directions=analysis_config.get("metric_directions", {}),
        # Manual audit-result ingestion is intentionally prohibited. Until a
        # validated deterministic runner is implemented, both mandatory gates
        # fail closed and the diagnostic can never become locked evidence.
        negative_control_passed=False,
        sensitivity_analysis_passed=False,
        decision_time_column=analysis_config.get("decision_time_column", "decision_time"),
        confounder_availability_suffix=analysis_config.get(
            "confounder_availability_suffix", "_available_at"
        ),
    )
    if not result.abstained:
        raise RuntimeError("causal audit gates must fail closed in this MVP")
    if args.output:
        _write_json(
            privacy.resolve_derived(args.output, create_parent=True),
            {
                "diagnostic": result,
                "locked_result": None,
                "registration_allowed": False,
            },
        )
    _emit(
        {
            "status": "abstained" if result.abstained else "estimated",
            "target_trial_spec_id": result.target_trial_spec_id,
            "target_trial_result_id": None,
            "eligible_patient_count": result.admission.eligible_patient_count,
            "effect_count": len(result.effects),
            "pareto_option_count": len(result.pareto_options),
            "abstention_reasons": result.abstention_reasons,
            "research_only": True,
        }
    )
    return 2


def _store(args: argparse.Namespace) -> ResearchStore:
    return ResearchStore(args.db, privacy=_privacy())


def _require_signed_record(
    store: ResearchStore, object_type: str, object_id: str
) -> dict[str, Any]:
    record = store.get(object_type, object_id)
    if record is None:
        raise ValueError(f"required {object_type} artifact not found")
    if not store.artifact_has_valid_signature(object_type, object_id):
        raise ValueError(f"required {object_type} artifact is not signature-bound")
    return record


def _register_state_from_store(
    registry: GuidelineRegistry, store: ResearchStore, state_id: str
) -> None:
    if registry.has_candidate_state(state_id):
        return
    state = CandidateDiseaseState.model_validate(
        _require_signed_record(store, "candidate_state", state_id)["payload"]
    )
    if not state.validation_report_id or not state.model_release_id:
        raise ValueError("candidate state evidence references are incomplete")
    report = StateValidationReport.model_validate(
        _require_signed_record(
            store, "state_validation_report", state.validation_report_id
        )["payload"]
    )
    model = ModelRelease.model_validate(
        _require_signed_record(store, "model_release", state.model_release_id)[
            "payload"
        ]
    )
    state_reviews = [
        ExpertReview.model_validate(
            _require_signed_record(store, "expert_review", review_id)["payload"]
        )
        for review_id in state.expert_review_ids
    ]
    model_reviews = [
        ExpertReview.model_validate(
            _require_signed_record(store, "expert_review", review_id)["payload"]
        )
        for review_id in model.expert_review_ids
    ]
    registry.register_candidate_state(
        state,
        model_release=model,
        validation_report=report,
        state_reviews=state_reviews,
        model_reviews=model_reviews,
    )


def _register_causal_result_from_store(
    registry: GuidelineRegistry, store: ResearchStore, result_id: str
) -> None:
    if registry.has_causal_result(result_id):
        return
    result = TargetTrialResult.model_validate(
        _require_signed_record(store, "target_trial_result", result_id)["payload"]
    )
    registry.register_causal_result(result)


def _guideline_build(args: argparse.Namespace) -> int:
    privacy = _privacy()
    document = _read_json(privacy.resolve_derived(args.input))
    if "candidate_states" in document or "causal_effects" in document:
        raise ValueError("candidate evidence must be loaded from the signed audit store")
    store = _store(args)
    if not store.verify_audit_chain():
        raise ValueError("audit chain verification failed")
    registry = GuidelineRegistry()
    recommendations: list[GuidelineRecommendation] = []
    for item in document["recommendations"]:
        recommendation = GuidelineRecommendation.model_validate(item)
        for state_id in recommendation.candidate_state_ids:
            _register_state_from_store(registry, store, state_id)
        for option in recommendation.treatment_options:
            for result_id in option.target_trial_result_ids:
                _register_causal_result_from_store(registry, store, result_id)
        recommendations.append(registry.register_recommendation(recommendation))
    release_config = document["release"]
    release = registry.create_release(
        name=release_config["name"],
        version=release_config["version"],
        channel=GuidelineChannel(release_config["channel"]),
        recommendation_ids=[item.id for item in recommendations],
        context_of_use=release_config["context_of_use"],
    )
    identity = GovernanceDirectory.from_env().authenticate(
        os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        allowed_roles=[
            "research_governance",
            "expert_committee",
            "crc_clinical_expert",
            "methodologist",
        ],
    )
    governance_token = os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN")
    for recommendation in recommendations:
        store.put(
            "guideline_recommendation",
            recommendation.id,
            recommendation,
            actor=identity.subject,
            governance_token=governance_token,
        )
        fhir = export_cpg_on_fhir(recommendation).as_bundle()
        store.put("fhir_cpg_bundle", recommendation.id, fhir)
    store.put(
        "guideline_release",
        release.id,
        release,
        actor=identity.subject,
        governance_token=governance_token,
    )
    _emit(
        {
            "status": release.status,
            "release_id": release.id,
            "channel": release.channel,
            "recommendation_count": len(recommendations),
            "automatic_orders": False,
        }
    )
    return 0


def _restore_registry(store: ResearchStore, release_id: str) -> GuidelineRegistry:
    if not store.verify_audit_chain():
        raise ValueError("audit chain verification failed")
    release_record = store.get("guideline_release", release_id)
    if release_record is None:
        raise ValueError("guideline release not found")
    release = GuidelineRelease.model_validate(release_record["payload"])
    registry = GuidelineRegistry()
    for recommendation_id in release.recommendation_ids:
        record = store.get("guideline_recommendation", recommendation_id)
        if record is None:
            raise ValueError("bound recommendation not found")
        recommendation = GuidelineRecommendation.model_validate(record["payload"])
        for state_id in recommendation.candidate_state_ids:
            _register_state_from_store(registry, store, state_id)
        for option in recommendation.treatment_options:
            for result_id in option.target_trial_result_ids:
                _register_causal_result_from_store(registry, store, result_id)
        registry.register_recommendation(recommendation)
    registry.restore_release(release)
    for record in store.list_all_latest("expert_review"):
        review = ExpertReview.model_validate(record["payload"])
        if review.object_id == release_id:
            registry.submit_review(review)
    return registry


def _guideline_lock(args: argparse.Namespace) -> int:
    store = _store(args)
    registry = _restore_registry(store, args.release_id)
    identity = GovernanceDirectory.from_env().authenticate(
        os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        allowed_roles=["research_governance", "expert_committee"],
    )
    release = registry.get_release(args.release_id)
    if release.status == "in_review":
        release = registry.lock_release(args.release_id)
        store.put(
            "guideline_release",
            release.id,
            release,
            actor=identity.subject,
            governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        )
    if args.publish:
        publisher_role = (
            "expert_committee"
            if "expert_committee" in identity.roles
            else "research_governance"
        )
        release = registry.publish_release(
            args.release_id, publisher_role=publisher_role
        )
        store.put(
            "guideline_release",
            release.id,
            release,
            actor=identity.subject,
            governance_token=os.environ.get("AI_CLINICIAN_GOVERNANCE_TOKEN"),
        )
    _emit(
        {
            "status": release.status,
            "release_id": release.id,
            "channel": release.channel,
            "review_count": len(release.expert_review_ids),
            "research_only": release.research_only,
            "automatic_orders": False,
        }
    )
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("research API may bind only to a loopback host")
    uvicorn.run(create_app(_store(args)), host=args.host, port=args.port)
    return 0


class _SafeArgumentParser(argparse.ArgumentParser):
    """Argparse variant that never reflects rejected argument values to stderr."""

    def error(self, message: str) -> None:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid command-line arguments\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="ai-clinician",
        description="Research-only mCRC timeline and computable-guideline platform",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    profile = data_commands.add_parser("profile")
    profile.add_argument("--workbook", action="append", required=True)
    profile.add_argument("--id-column", action="append")
    profile.add_argument("--header-rows", type=int, default=4)
    profile.add_argument("--output")
    profile.add_argument("--register", action="store_true")
    profile.add_argument("--db", default="metadata/ai_clinician.db")
    profile.set_defaults(handler=_data_profile)
    link = data_commands.add_parser("link")
    link.add_argument("--workbook", action="append", required=True)
    link.add_argument("--id-column")
    link.add_argument("--field-aliases")
    link.add_argument("--header-rows", type=int, default=4)
    link.add_argument("--output", required=True)
    link.set_defaults(handler=_data_link)
    cohort_lock = data_commands.add_parser("lock-cohort")
    cohort_lock.add_argument("--input", required=True)
    cohort_lock.add_argument("--db", default="metadata/ai_clinician.db")
    cohort_lock.set_defaults(handler=_data_lock_cohort)

    timeline = commands.add_parser("timeline")
    timeline_commands = timeline.add_subparsers(dest="timeline_command", required=True)
    timeline_build = timeline_commands.add_parser("build")
    timeline_build.add_argument("--input", required=True)
    timeline_build.add_argument("--output", required=True)
    timeline_build.set_defaults(handler=_timeline_build)
    timeline_register = timeline_commands.add_parser("register-manifest")
    timeline_register.add_argument("--gold", required=True)
    timeline_register.add_argument("--decision-manifest", required=True)
    timeline_register.add_argument("--db", default="metadata/ai_clinician.db")
    timeline_register.set_defaults(handler=_timeline_register_manifest)
    timeline_validate = timeline_commands.add_parser("validate")
    timeline_validate.add_argument("--predicted", required=True)
    timeline_validate.add_argument("--gold", required=True)
    timeline_validate.add_argument(
        "--decision-manifest", "--decision-times", dest="decision_manifest", required=True
    )
    timeline_validate.add_argument("--output")
    timeline_validate.add_argument("--register", action="store_true")
    timeline_validate.add_argument("--manifest-registration-id", required=True)
    timeline_validate.add_argument("--db", default="metadata/ai_clinician.db")
    timeline_validate.set_defaults(handler=_timeline_validate)

    states = commands.add_parser("states")
    state_commands = states.add_subparsers(dest="states_command", required=True)
    state_register_manifest = state_commands.add_parser("register-manifest")
    state_register_manifest.add_argument("--input", required=True)
    state_register_manifest.add_argument("--db", default="metadata/ai_clinician.db")
    state_register_manifest.set_defaults(handler=_states_register_manifest)
    state_fit = state_commands.add_parser("fit")
    state_fit.add_argument("--input", required=True)
    state_fit.add_argument("--output", required=True)
    state_fit.add_argument("--db", default="metadata/ai_clinician.db")
    state_fit.set_defaults(handler=_states_fit)
    state_evaluate = state_commands.add_parser("evaluate")
    state_evaluate.add_argument("--input", required=True)
    state_evaluate.add_argument("--output")
    state_evaluate.add_argument("--register", action="store_true")
    state_evaluate.add_argument("--db", default="metadata/ai_clinician.db")
    state_evaluate.set_defaults(handler=_states_evaluate)

    trial = commands.add_parser("target-trial")
    trial_commands = trial.add_subparsers(dest="trial_command", required=True)
    trial_register_spec = trial_commands.add_parser("register-spec")
    trial_register_spec.add_argument("--input", required=True)
    trial_register_spec.add_argument("--db", default="metadata/ai_clinician.db")
    trial_register_spec.set_defaults(handler=_target_trial_register_spec)
    trial_freeze_input = trial_commands.add_parser("freeze-input")
    trial_freeze_input.add_argument("--input", required=True)
    trial_freeze_input.add_argument("--db", default="metadata/ai_clinician.db")
    trial_freeze_input.set_defaults(handler=_target_trial_freeze_input)
    trial_run = trial_commands.add_parser("run")
    trial_run.add_argument("--input", required=True)
    trial_run.add_argument("--output")
    trial_run.add_argument("--register", action="store_true")
    trial_run.add_argument("--db", default="metadata/ai_clinician.db")
    trial_run.set_defaults(handler=_target_trial_run)

    guideline = commands.add_parser("guideline")
    guideline_commands = guideline.add_subparsers(dest="guideline_command", required=True)
    guideline_build = guideline_commands.add_parser("build")
    guideline_build.add_argument("--input", required=True)
    guideline_build.add_argument("--db", default="metadata/ai_clinician.db")
    guideline_build.set_defaults(handler=_guideline_build)
    guideline_lock = guideline_commands.add_parser("lock")
    guideline_lock.add_argument("--release-id", required=True)
    guideline_lock.add_argument("--db", default="metadata/ai_clinician.db")
    guideline_lock.add_argument("--publish", action="store_true")
    guideline_lock.set_defaults(handler=_guideline_lock)

    serve = commands.add_parser("serve")
    serve.add_argument("--db", default="metadata/ai_clinician.db")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        # Do not echo exception messages: parser/model errors can contain patient text.
        _emit(
            {"status": "error", "error_type": type(exc).__name__},
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
