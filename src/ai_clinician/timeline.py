"""Deterministic, provenance-preserving Chinese clinical timeline extraction.

The rule extractor is intentionally conservative.  A pluggable extractor may
add locally produced events, but network-backed extractors are rejected and no
cloud client is imported by this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable

from .models import ClinicalEvent, EvidenceSpan, PatientTimeline


UTC = timezone.utc


@runtime_checkable
class LocalClinicalExtractor(Protocol):
    """Protocol for an optional on-premises entity/relation extractor.

    Implementations must declare ``execution_mode = "local"``.  This is an
    explicit integration guard rather than an assertion that arbitrary plugin
    code is safe; deployments must still review and sandbox model adapters.
    """

    execution_mode: Literal["local"]

    def extract(
        self,
        *,
        patient_key: str,
        text: str,
        source_document_id: str,
        recorded_at: datetime | None,
        sheet_name: str | None = None,
        cell_ref: str | None = None,
    ) -> Iterable[ClinicalEvent | Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class _DateMention:
    value: datetime
    precision: str
    start: int
    end: int


_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>19\d{2}|20\d{2})"
    r"(?:\s*[年./-]\s*(?P<month>0?[1-9]|1[0-2])"
    r"(?:\s*[月./-]\s*(?P<day>0?[1-9]|[12]\d|3[01])\s*日?)?"
    r"|\s*年)?(?!\d)"
)

_LINE_RE = re.compile(
    r"(?:第)?(?P<cn>[一二三四五六七八九十]+)线|"
    r"(?:第)?(?P<num>[1-9]\d*)线|(?P<short>[1-9])\s*L",
    re.IGNORECASE,
)

_REGIMENS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"m?FOLFOXIRI", re.IGNORECASE), "FOLFOXIRI"),
    (re.compile(r"m?FOLFOX(?:\s*[- ]?\d+)?", re.IGNORECASE), "FOLFOX"),
    (re.compile(r"FOLFIRI", re.IGNORECASE), "FOLFIRI"),
    (re.compile(r"CAPOX|XELOX", re.IGNORECASE), "CAPOX/XELOX"),
    (re.compile(r"TAS\s*[- ]?102", re.IGNORECASE), "TAS-102"),
    (re.compile(r"奥沙利铂"), "奥沙利铂"),
    (re.compile(r"伊立替康"), "伊立替康"),
    (re.compile(r"卡培他滨"), "卡培他滨"),
    (re.compile(r"5\s*[- ]?FU|氟尿嘧啶", re.IGNORECASE), "氟尿嘧啶"),
    (re.compile(r"贝伐珠单抗"), "贝伐珠单抗"),
    (re.compile(r"西妥昔单抗"), "西妥昔单抗"),
    (re.compile(r"帕尼单抗"), "帕尼单抗"),
    (re.compile(r"雷替曲塞"), "雷替曲塞"),
    (re.compile(r"曲氟尿苷替匹嘧啶"), "曲氟尿苷替匹嘧啶"),
    (re.compile(r"呋喹替尼"), "呋喹替尼"),
    (re.compile(r"瑞戈非尼"), "瑞戈非尼"),
)

_PROGRESSION_RE = re.compile(
    r"疾病进展|病情进展|肿瘤进展|影像学进展|新发病灶|复发|"
    r"(?<![A-Za-z])PD(?![A-Za-z])",
    re.IGNORECASE,
)
_METASTASIS_RE = re.compile(
    r"(?P<site>肝|肺|腹膜|骨|脑|淋巴结|卵巢|远处|多发)?转移(?:灶|性病变)?"
)
_DEATH_RE = re.compile(r"死亡|病逝|去世")
_ECOG_RE = re.compile(r"ECOG(?:\s*PS|评分|体力状态)?\s*[:：为]?\s*([0-5])\s*分?", re.IGNORECASE)
_DIAGNOSIS_RE = re.compile(r"(?:诊断为|确诊(?:为)?)\s*(结直肠癌|结肠癌|直肠癌)")
_STAGE_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:yp|p|c)?T[0-4x][a-d]?\s*N[0-3x][a-c]?\s*M[0-1x][a-c]?)",
    re.IGNORECASE,
)
_SURGERY_RE = re.compile(
    r"结肠切除术|直肠切除术|根治性切除术|根治术|转移灶切除术|姑息性手术|手术切除"
)
_RESPONSE_RE = re.compile(
    r"完全缓解|部分缓解|疾病稳定|病情稳定|(?<![A-Za-z])(?:CR|PR|SD)(?![A-Za-z])",
    re.IGNORECASE,
)
_FOLLOW_UP_RE = re.compile(r"随访|末次就诊")
_MOLECULAR_RE = re.compile(
    r"(?P<marker>KRAS|NRAS|BRAF|HER2|MSI|MMR)\s*"
    r"(?P<result>野生型|突变型|突变|扩增|阳性|阴性|高|低|稳定|缺失|正常)",
    re.IGNORECASE,
)
_TOXICITY_NAMES: tuple[str, ...] = (
    "中性粒细胞减少",
    "白细胞减少",
    "血小板减少",
    "手足综合征",
    "周围神经毒性",
    "神经毒性",
    "肝功能损伤",
    "不良反应",
    "蛋白尿",
    "高血压",
    "贫血",
    "恶心",
    "呕吐",
    "腹泻",
    "皮疹",
    "毒性",
)
_GRADE_RE = re.compile(
    r"(?:(?:CTCAE\s*)?G(?:rade)?\s*([1-5])|"
    r"([1-5]|I{1,3}|IV|V|[一二三四五])\s*[级度])",
    re.IGNORECASE,
)
_TREATMENT_END_RE = re.compile(r"停用|停止|终止|结束|完成(?:该|此)?方案")

_SINGLE_VALUE_EVENT_TYPES = {
    "diagnosis",
    "ecog",
    "pathology_stage",
    "molecular_test",
    "response",
    "treatment_start",
    "treatment_end",
}

_NEGATION_RE = re.compile(
    r"(?:未见|未发现|未发生|未出现|无|否认|排除|不考虑|未)"
    r"(?:任何|明显|明确|新发)?[^，,。；;！？\n]{0,10}$"
)


def _as_utc(value: datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _date_mentions(text: str) -> tuple[_DateMention, ...]:
    mentions: list[_DateMention] = []
    for match in _DATE_RE.finditer(text):
        year = int(match.group("year"))
        month_text = match.group("month")
        day_text = match.group("day")
        month = int(month_text) if month_text else 1
        day = int(day_text) if day_text else 1
        precision = "day" if day_text else "month" if month_text else "year"
        try:
            parsed = datetime(year, month, day, tzinfo=UTC)
        except ValueError:
            continue
        mentions.append(
            _DateMention(parsed, precision, match.start(), match.end())
        )
    return tuple(mentions)


def _nearest_date(
    mentions: Sequence[_DateMention], position: int, segment_start: int, segment_end: int
) -> _DateMention | None:
    local = [m for m in mentions if m.start < segment_end and m.end > segment_start]
    if local:
        prior = [m for m in local if m.start <= position]
        return max(prior, key=lambda m: m.start) if prior else min(
            local, key=lambda m: abs(m.start - position)
        )
    # A date in the immediately preceding short clause is a common Chinese note
    # form ("2023年1月。开始一线...").  Do not carry dates across long passages.
    prior = [m for m in mentions if m.end <= position and position - m.end <= 24]
    return max(prior, key=lambda m: m.end) if prior else None


def _segments(text: str) -> Iterable[tuple[int, int, str]]:
    for match in re.finditer(r"[^。！？；;\n]+", text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        if start < end:
            yield start, end, text[start:end]


def _is_negated(segment: str, match_start: int) -> bool:
    prefix = segment[max(0, match_start - 20) : match_start]
    # Reset the scope at punctuation or contrast conjunctions.
    prefix = re.split(r"[，,。；;！？\n]|但|然而|后", prefix)[-1]
    return bool(_NEGATION_RE.search(prefix))


def _chinese_number(text: str) -> int:
    if text.isdigit():
        return int(text)
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        return (digits.get(left, 1) * 10) + digits.get(right, 0)
    return digits[text]


def _line_number(match: re.Match[str] | None) -> int | None:
    if match is None:
        return None
    if match.group("cn"):
        return _chinese_number(match.group("cn"))
    return int(match.group("num") or match.group("short"))


def _grade_number(value: str) -> int:
    value = value.upper()
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    chinese = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    if value.isdigit():
        return int(value)
    if value in roman:
        return roman[value]
    return chinese[value]


def _stable_event_id(
    patient_key: str,
    source_document_id: str,
    event_type: str,
    start: int,
    end: int,
    event_time: datetime | None,
    code: str | None,
    value: str | int | float | bool | None,
) -> str:
    material = json.dumps(
        [
            patient_key,
            source_document_id,
            event_type,
            start,
            end,
            event_time.isoformat() if event_time else None,
            code,
            value,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"evt_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def _evidence_span(
    text: str,
    *,
    start: int,
    end: int,
    source_document_id: str,
    sheet_name: str | None,
    cell_ref: str | None,
) -> EvidenceSpan:
    evidence = text[start:end]
    return EvidenceSpan(
        source_document_id=source_document_id,
        sheet_name=sheet_name,
        cell_ref=cell_ref,
        start=start,
        end=end,
        text_sha256=hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
    )


def _make_event(
    *,
    patient_key: str,
    event_type: str,
    text: str,
    source_document_id: str,
    segment_start: int,
    segment_end: int,
    event_position: int,
    mentions: Sequence[_DateMention],
    recorded_at: datetime | None,
    confidence: float,
    code: str | None = None,
    value: str | int | float | bool | None = None,
    unit: str | None = None,
    sheet_name: str | None = None,
    cell_ref: str | None = None,
) -> ClinicalEvent:
    date_mention = _nearest_date(mentions, event_position, segment_start, segment_end)
    event_time = date_mention.value if date_mention else None
    missing_flags: list[str] = []
    if event_time is None:
        missing_flags.append("event_time_missing")
    if recorded_at is None:
        missing_flags.append("source_available_at_missing")
    return ClinicalEvent(
        id=_stable_event_id(
            patient_key,
            source_document_id,
            event_type,
            segment_start,
            segment_end,
            event_time,
            code,
            value,
        ),
        patient_key=patient_key,
        event_type=event_type,
        event_time=event_time,
        time_precision=date_mention.precision if date_mention else "unknown",
        code=code,
        value=value,
        unit=unit,
        confidence=confidence,
        evidence_spans=[
            _evidence_span(
                text,
                start=segment_start,
                end=segment_end,
                source_document_id=source_document_id,
                sheet_name=sheet_name,
                cell_ref=cell_ref,
            )
        ],
        source_available_at=recorded_at,
        conflicts=[],
        missing_flags=missing_flags,
    )


def _event_type_value(event: ClinicalEvent) -> str:
    value = event.event_type
    return str(getattr(value, "value", value))


def _extract_rule_events(
    *,
    patient_key: str,
    text: str,
    source_document_id: str,
    recorded_at: datetime | None,
    sheet_name: str | None,
    cell_ref: str | None,
) -> list[ClinicalEvent]:
    mentions = _date_mentions(text)
    events: list[ClinicalEvent] = []

    def emit(
        event_type: str,
        *,
        segment_start: int,
        segment_end: int,
        position: int,
        confidence: float,
        code: str | None = None,
        value: str | int | float | bool | None = None,
        unit: str | None = None,
    ) -> None:
        events.append(
            _make_event(
                patient_key=patient_key,
                event_type=event_type,
                text=text,
                source_document_id=source_document_id,
                segment_start=segment_start,
                segment_end=segment_end,
                event_position=position,
                mentions=mentions,
                recorded_at=recorded_at,
                confidence=confidence,
                code=code,
                value=value,
                unit=unit,
                sheet_name=sheet_name,
                cell_ref=cell_ref,
            )
        )

    for segment_start, segment_end, segment in _segments(text):
        # Diagnosis, stage, molecular tests, surgery and response are included so
        # the standard layer can represent the full mCRC path, even though the
        # first validation gate focuses on line, progression and death.
        for match in _DIAGNOSIS_RE.finditer(segment):
            if not _is_negated(segment, match.start()):
                emit(
                    "diagnosis",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=segment_start + match.start(),
                    confidence=0.97,
                    value=match.group(1),
                )

        for match in _STAGE_RE.finditer(segment):
            emit(
                "pathology_stage",
                segment_start=segment_start,
                segment_end=segment_end,
                position=segment_start + match.start(),
                confidence=0.98,
                code="TNM",
                value=re.sub(r"\s+", "", match.group(1)).upper(),
            )

        for match in _MOLECULAR_RE.finditer(segment):
            if not _is_negated(segment, match.start()):
                emit(
                    "molecular_test",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=segment_start + match.start(),
                    confidence=0.97,
                    code=match.group("marker").upper(),
                    value=match.group("result"),
                )

        for match in _SURGERY_RE.finditer(segment):
            if not _is_negated(segment, match.start()):
                emit(
                    "surgery",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=segment_start + match.start(),
                    confidence=0.96,
                    value=match.group(0),
                )

        line_matches = list(_LINE_RE.finditer(segment))
        regimen_hits: list[tuple[int, str]] = []
        occupied: list[tuple[int, int]] = []
        for pattern, canonical in _REGIMENS:
            for match in pattern.finditer(segment):
                if any(match.start() < end and match.end() > start for start, end in occupied):
                    continue
                regimen_hits.append((match.start(), canonical))
                occupied.append((match.start(), match.end()))
        if line_matches:
            assigned: dict[int, list[tuple[int, str]]] = defaultdict(list)
            for hit in regimen_hits:
                nearest_line = min(
                    range(len(line_matches)),
                    key=lambda index: abs(hit[0] - line_matches[index].start()),
                )
                assigned[nearest_line].append(hit)
            positions = [match.start() for match in line_matches]
            boundaries = [0] + [
                (positions[index - 1] + positions[index]) // 2
                for index in range(1, len(positions))
            ] + [len(segment)]
            for index, line_match in enumerate(line_matches):
                line = _line_number(line_match)
                position = segment_start + line_match.start()
                if _is_negated(segment, line_match.start()):
                    continue
                regimens = list(
                    dict.fromkeys(name for _, name in sorted(assigned.get(index, [])))
                )
                local_context = segment[boundaries[index] : boundaries[index + 1]]
                event_type = (
                    "treatment_end"
                    if _TREATMENT_END_RE.search(local_context)
                    else "treatment_start"
                )
                emit(
                    event_type,
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=position,
                    confidence=0.98 if line is not None and regimens else 0.92,
                    code=f"treatment_line:{line}" if line is not None else None,
                    value="+".join(regimens) if regimens else f"第{line}线治疗",
                )
        elif regimen_hits:
            position = segment_start + min(hit[0] for hit in regimen_hits)
            if not _is_negated(segment, position - segment_start):
                regimens = list(dict.fromkeys(name for _, name in sorted(regimen_hits)))
                emit(
                    "treatment_end" if _TREATMENT_END_RE.search(segment) else "treatment_start",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=position,
                    confidence=0.92,
                    value="+".join(regimens),
                )

        for match in _PROGRESSION_RE.finditer(segment):
            if not _is_negated(segment, match.start()):
                emit(
                    "progression",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=segment_start + match.start(),
                    confidence=0.96,
                    value=match.group(0).upper() if match.group(0).isascii() else match.group(0),
                )

        for match in _METASTASIS_RE.finditer(segment):
            if not _is_negated(segment, match.start()):
                site = match.group("site") or "未特指"
                emit(
                    "metastasis",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=segment_start + match.start(),
                    confidence=0.95 if site != "未特指" else 0.90,
                    code="metastatic_site",
                    value=site,
                )

        for match in _DEATH_RE.finditer(segment):
            if not _is_negated(segment, match.start()):
                emit(
                    "death",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=segment_start + match.start(),
                    confidence=0.99,
                    value=True,
                )

        for match in _ECOG_RE.finditer(segment):
            if not _is_negated(segment, match.start()):
                emit(
                    "ecog",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=segment_start + match.start(),
                    confidence=0.99,
                    code="ECOG",
                    value=int(match.group(1)),
                    unit="score",
                )

        for match in _RESPONSE_RE.finditer(segment):
            if not _is_negated(segment, match.start()):
                response = match.group(0).upper() if match.group(0).isascii() else match.group(0)
                emit(
                    "response",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=segment_start + match.start(),
                    confidence=0.95,
                    value=response,
                )

        for match in _FOLLOW_UP_RE.finditer(segment):
            emit(
                "follow_up",
                segment_start=segment_start,
                segment_end=segment_end,
                position=segment_start + match.start(),
                confidence=0.92,
                value=match.group(0),
            )

        toxicity_spans: list[tuple[int, int]] = []
        for name in _TOXICITY_NAMES:
            for match in re.finditer(re.escape(name), segment):
                if any(match.start() < end and match.end() > start for start, end in toxicity_spans):
                    continue
                if _is_negated(segment, match.start()):
                    continue
                toxicity_spans.append((match.start(), match.end()))
                local_start = max(0, match.start() - 12)
                local_end = min(len(segment), match.end() + 12)
                grade_candidates = list(_GRADE_RE.finditer(segment[local_start:local_end]))
                grade: int | None = None
                if grade_candidates:
                    nearest = min(
                        grade_candidates,
                        key=lambda item: abs((local_start + item.start()) - match.start()),
                    )
                    grade = _grade_number(nearest.group(1) or nearest.group(2))
                emit(
                    "toxicity",
                    segment_start=segment_start,
                    segment_end=segment_end,
                    position=segment_start + match.start(),
                    confidence=0.97 if grade is not None else 0.88,
                    code=name,
                    value=grade,
                    unit="CTCAE grade" if grade is not None else None,
                )
    return events


def _deduplicate(events: Iterable[ClinicalEvent]) -> list[ClinicalEvent]:
    deduplicated: dict[tuple[Any, ...], ClinicalEvent] = {}
    for event in events:
        spans = tuple(
            (span.source_document_id, span.start, span.end, span.text_sha256)
            for span in event.evidence_spans
        )
        key = (
            _event_type_value(event),
            event.event_time,
            event.code,
            json.dumps(event.value, ensure_ascii=False, sort_keys=True),
            spans,
        )
        existing = deduplicated.get(key)
        if existing is None or event.confidence > existing.confidence:
            deduplicated[key] = event
    return list(deduplicated.values())


def _annotate_conflicts(
    events: Sequence[ClinicalEvent],
) -> tuple[list[ClinicalEvent], list[str]]:
    grouped: dict[tuple[str, datetime | None, str | None], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        if (
            event.event_time is not None
            and _event_type_value(event) in _SINGLE_VALUE_EVENT_TYPES
        ):
            grouped[(_event_type_value(event), event.event_time, event.code)].append(index)

    conflicting_indices: set[int] = set()
    conflicts: list[str] = []
    for (event_type, _, code), indices in grouped.items():
        values = {
            json.dumps(events[index].value, ensure_ascii=False, sort_keys=True)
            for index in indices
        }
        if len(values) > 1:
            conflicting_indices.update(indices)
            conflicts.append(f"conflicting_values_same_time:{event_type}:{code or 'uncoded'}")

    annotated: list[ClinicalEvent] = []
    for index, event in enumerate(events):
        if index not in conflicting_indices:
            annotated.append(event)
            continue
        event_conflicts = list(event.conflicts)
        if "conflicting_values_same_time" not in event_conflicts:
            event_conflicts.append("conflicting_values_same_time")
        annotated.append(event.model_copy(update={"conflicts": event_conflicts}))
    return annotated, sorted(set(conflicts))


def _sort_key(event: ClinicalEvent) -> tuple[datetime, datetime, str, str]:
    maximum = datetime.max.replace(tzinfo=UTC)
    return (
        _as_utc(event.event_time) or maximum,
        _as_utc(event.source_available_at) or maximum,
        _event_type_value(event),
        event.id,
    )


def _timeline_completeness(events: Sequence[ClinicalEvent]) -> float:
    if not events:
        return 0.0
    dated = sum(event.event_time is not None for event in events) / len(events)
    evidenced = sum(bool(event.evidence_spans) for event in events) / len(events)
    return round((dated + evidenced) / 2, 6)


def build_timeline(
    patient_key: str,
    text: str,
    *,
    source_document_id: str,
    recorded_at: datetime | date | None = None,
    sheet_name: str | None = None,
    cell_ref: str | None = None,
    local_extractor: LocalClinicalExtractor | None = None,
) -> PatientTimeline:
    """Build one provenance-preserving patient timeline from Chinese text."""

    if not patient_key.strip():
        raise ValueError("patient_key is required")
    if not source_document_id.strip():
        raise ValueError("source_document_id is required")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    available_at = _as_utc(recorded_at)
    events = _extract_rule_events(
        patient_key=patient_key,
        text=text,
        source_document_id=source_document_id,
        recorded_at=available_at,
        sheet_name=sheet_name,
        cell_ref=cell_ref,
    )

    if local_extractor is not None:
        if getattr(local_extractor, "execution_mode", None) != "local":
            raise ValueError("pluggable extractors must declare execution_mode='local'")
        extracted = local_extractor.extract(
            patient_key=patient_key,
            text=text,
            source_document_id=source_document_id,
            recorded_at=available_at,
            sheet_name=sheet_name,
            cell_ref=cell_ref,
        )
        for candidate in extracted:
            event = candidate if isinstance(candidate, ClinicalEvent) else ClinicalEvent.model_validate(candidate)
            if event.patient_key != patient_key:
                raise ValueError("local extractor returned an event for another patient")
            if not event.evidence_spans:
                raise ValueError("local extractor events require at least one evidence span")
            if _as_utc(event.source_available_at) != available_at:
                raise ValueError(
                    "local extractor source availability must equal the source record time"
                )
            for span in event.evidence_spans:
                if span.source_document_id != source_document_id:
                    raise ValueError(
                        "local extractor evidence must bind the current source document"
                    )
                if span.sheet_name != sheet_name or span.cell_ref != cell_ref:
                    raise ValueError(
                        "local extractor evidence must bind the current source location"
                    )
                if span.end > len(text):
                    raise ValueError("local extractor evidence span exceeds source text")
                evidence_hash = hashlib.sha256(
                    text[span.start : span.end].encode("utf-8")
                ).hexdigest()
                if evidence_hash != span.text_sha256:
                    raise ValueError(
                        "local extractor evidence hash does not match source text"
                    )
            events.append(event)

    events = _deduplicate(events)
    events, conflicts = _annotate_conflicts(events)
    events.sort(key=_sort_key)
    return PatientTimeline(
        patient_key=patient_key,
        events=events,
        completeness=_timeline_completeness(events),
        conflicts=conflicts,
        leakage_flags=[],
    )


class HybridTimelineExtractor:
    """Reusable deterministic rules plus an optional reviewed local model."""

    def __init__(self, local_extractor: LocalClinicalExtractor | None = None) -> None:
        if local_extractor is not None and getattr(local_extractor, "execution_mode", None) != "local":
            raise ValueError("pluggable extractors must declare execution_mode='local'")
        self._local_extractor = local_extractor

    def extract(
        self,
        patient_key: str,
        text: str,
        *,
        source_document_id: str,
        recorded_at: datetime | date | None = None,
        sheet_name: str | None = None,
        cell_ref: str | None = None,
    ) -> PatientTimeline:
        return build_timeline(
            patient_key,
            text,
            source_document_id=source_document_id,
            recorded_at=recorded_at,
            sheet_name=sheet_name,
            cell_ref=cell_ref,
            local_extractor=self._local_extractor,
        )


def leakage_reasons(
    timeline: PatientTimeline,
    decision_time: datetime | date,
    *,
    include_unknown_availability: bool = True,
) -> tuple[str, ...]:
    """Return aggregate-safe reasons why events are invalid at a decision time."""

    cutoff = _as_utc(decision_time)
    assert cutoff is not None
    reasons: list[str] = []
    for event in timeline.events:
        available = _as_utc(event.source_available_at)
        occurred = _as_utc(event.event_time)
        if available is None and include_unknown_availability:
            reasons.append(f"source_availability_unknown:{event.id}")
        elif available is not None and available > cutoff:
            reasons.append(f"future_source_evidence:{event.id}")
        if occurred is not None and occurred > cutoff:
            reasons.append(f"future_event_time:{event.id}")
    return tuple(reasons)


def decision_time_view(
    timeline: PatientTimeline,
    decision_time: datetime | date,
    *,
    allow_unknown_availability: bool = False,
) -> PatientTimeline:
    """Create a point-in-time view that cannot contain future evidence.

    Unknown source availability is excluded by default because its temporal
    eligibility cannot be proven.  The returned view has no leakage flags; use
    :func:`leakage_reasons` on an unfiltered timeline for an audit report.
    """

    cutoff = _as_utc(decision_time)
    assert cutoff is not None
    eligible: list[ClinicalEvent] = []
    for event in timeline.events:
        available = _as_utc(event.source_available_at)
        occurred = _as_utc(event.event_time)
        if available is None and not allow_unknown_availability:
            continue
        if available is not None and available > cutoff:
            continue
        if occurred is not None and occurred > cutoff:
            continue
        eligible.append(event)
    cleaned = [
        event.model_copy(
            update={
                "conflicts": [
                    conflict
                    for conflict in event.conflicts
                    if conflict != "conflicting_values_same_time"
                ]
            }
        )
        for event in eligible
    ]
    eligible, retained_conflicts = _annotate_conflicts(cleaned)
    return timeline.model_copy(
        update={
            "events": eligible,
            "completeness": _timeline_completeness(eligible),
            "conflicts": retained_conflicts,
            "leakage_flags": [],
        }
    )


# Backward-compatible, CLI-friendly aliases with explicit verbs.
extract_timeline = build_timeline
validate_no_future_leakage = leakage_reasons
