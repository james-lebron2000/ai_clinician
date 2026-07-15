from __future__ import annotations

import hashlib
import hmac
from collections import Counter
from dataclasses import asdict, fields

import pytest

from ai_clinician.data import (
    QuarantinedRow,
    WorkbookProfileError,
    link_patient_views,
)


openpyxl = pytest.importorskip("openpyxl")
SECRET = b"synthetic-linkage-secret-v1"


def _write_workbook(path, *, title, headers, rows):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = title
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patient_key(identifier):
    digest = hmac.new(
        SECRET,
        b"patient-key-v1\0" + identifier.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"pt_{digest}"


def test_linkage_quarantines_bad_keys_and_retains_provenance_and_conflicts(
    tmp_path,
):
    clinical = tmp_path / "clinical.xlsx"
    molecular = tmp_path / "molecular.xlsx"
    _write_workbook(
        clinical,
        title="临床合成表",
        headers=["患者编号", "治疗方案", "ECOG", "姓名"],
        rows=[
            ["SYN-001", "FOLFOX", 1, "合成人甲"],
            ["SYN-002", "FOLFIRI", 2, "合成人乙"],
            ["SYN-002", "FOLFOX", 1, "合成人乙"],
            [None, "CAPOX", 0, "合成人丙"],
            ["SYN-003", "CAPOX", 0, "合成人丁"],
        ],
    )
    _write_workbook(
        molecular,
        title="分子合成表",
        headers=["patient_id", "治疗方案", "KRAS"],
        rows=[
            ["SYN-001", "FOLFIRI", "突变"],
            ["SYN-003", "CAPOX", "野生型"],
            ["SYN-004", "FOLFOX", "野生型"],
        ],
    )
    source_hashes = {path: _file_hash(path) for path in (clinical, molecular)}
    files_before = set(tmp_path.iterdir())

    result = link_patient_views(
        [clinical, molecular], hmac_secret=SECRET, header_rows=1
    )

    assert result.source_document_count == 2
    assert result.sheet_count == 2
    assert len(result.patient_views) == 3
    assert {view.patient_key for view in result.patient_views} == {
        _patient_key("SYN-001"),
        _patient_key("SYN-003"),
        _patient_key("SYN-004"),
    }

    # Every row in the duplicate group is rejected; no positional winner is
    # silently selected.  The missing key has its own non-identifying pointer.
    reasons = Counter(row.reason for row in result.quarantine)
    assert reasons == {"duplicate_identifier": 2, "missing_identifier": 1}
    assert tuple(field.name for field in fields(QuarantinedRow)) == (
        "source_document_id",
        "sheet_name",
        "reason",
        "row_hash",
    )
    assert all(row.row_hash.startswith("row_") for row in result.quarantine)

    patient = next(
        view for view in result.patient_views if view.patient_key == _patient_key("SYN-001")
    )
    conflict = next(
        item for item in patient.conflicts if item.field_name == "治疗方案"
    )
    assert conflict.distinct_value_count == 2
    conflicting = [patient.observations[index] for index in conflict.observation_indexes]
    assert {item.value for item in conflicting} == {"FOLFOX", "FOLFIRI"}
    assert {item.source.cell_ref for item in conflicting} == {"B2"}
    assert all(item.source.source_document_id.startswith("xlsx_") for item in conflicting)
    assert {item.source.sheet_name for item in conflicting} == {
        "临床合成表",
        "分子合成表",
    }

    # The result, including dataclass serialisation, contains no raw primary key.
    serialised = repr(result) + repr(asdict(result))
    for raw_identifier in ("SYN-001", "SYN-002", "SYN-003", "SYN-004"):
        assert raw_identifier not in serialised
    assert "合成人甲" not in serialised

    # The API is read-only and has no output path: no file was changed or added.
    assert set(tmp_path.iterdir()) == files_before
    assert {path: _file_hash(path) for path in (clinical, molecular)} == source_hashes


def test_explicit_identifier_header_and_field_aliases(tmp_path):
    baseline = tmp_path / "baseline.xlsx"
    follow_up = tmp_path / "follow-up.xlsx"
    _write_workbook(
        baseline,
        title="基线",
        headers=["研究编码", "治疗方案"],
        rows=[["CASE-A", "FOLFOX"]],
    )
    _write_workbook(
        follow_up,
        title="随访",
        headers=["研究编码", "方案"],
        rows=[["CASE-A", "FOLFIRI"]],
    )

    result = link_patient_views(
        [baseline, follow_up],
        hmac_secret=SECRET,
        identifier_header="研究编码",
        identifier_aliases=(),
        field_aliases={"方案": "治疗方案"},
        header_rows=1,
    )

    assert len(result.patient_views) == 1
    patient = result.patient_views[0]
    assert patient.patient_key == _patient_key("CASE-A")
    assert [conflict.field_name for conflict in patient.conflicts] == ["治疗方案"]
    assert "CASE-A" not in repr(result)


def test_missing_identifier_column_is_quarantined_without_source_text(tmp_path):
    source = tmp_path / "missing-key-column.xlsx"
    _write_workbook(
        source,
        title="无主键",
        headers=["治疗方案", "ECOG"],
        rows=[["FOLFOX", 1], ["CAPOX", 0]],
    )

    result = link_patient_views(
        [source], hmac_secret=SECRET, identifier_aliases=(), header_rows=1
    )

    assert result.patient_views == ()
    assert [row.reason for row in result.quarantine] == [
        "identifier_column_not_found",
        "identifier_column_not_found",
    ]
    assert "FOLFOX" not in repr(result)
    assert "CAPOX" not in repr(result)


def test_linkage_rejects_short_hmac_secret(tmp_path):
    source = tmp_path / "synthetic.xlsx"
    _write_workbook(
        source,
        title="合成",
        headers=["patient_id", "ECOG"],
        rows=[["SYN-001", 1]],
    )

    with pytest.raises(WorkbookProfileError, match="at least 16 bytes"):
        link_patient_views([source], hmac_secret=b"too-short", header_rows=1)
    with pytest.raises(WorkbookProfileError, match="placeholder"):
        link_patient_views(
            [source],
            hmac_secret="replace-with-an-approved-random-secret",
            header_rows=1,
        )
