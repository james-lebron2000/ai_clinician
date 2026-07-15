from __future__ import annotations

import hashlib

import pytest

from ai_clinician.data import (
    WorkbookProfileError,
    flatten_headers,
    profile_to_dict,
    profile_workbook,
    profile_workbooks,
)


openpyxl = pytest.importorskip("openpyxl")


def _synthetic_workbook(path, identifiers):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "合成队列"
    sheet.append(["基本信息", None, "病程"])
    sheet.append(["患者", None, "治疗"])
    sheet.append(["标识", None, "方案"])
    sheet.append(["患者ID", "合成姓名", "方案"])
    for index, identifier in enumerate(identifiers, start=1):
        sheet.append([identifier, f"SYNTHETIC-{index}", "FOLFOX"])
    workbook.save(path)
    workbook.close()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_flatten_four_header_rows_is_stable_and_unique():
    rows = [
        ["基本信息", None, "病程"],
        ["患者", None, "治疗"],
        ["标识", None, "方案"],
        ["患者ID", "患者ID", "方案"],
    ]

    assert flatten_headers(rows) == (
        "基本信息__患者__标识__患者ID",
        "基本信息__患者__标识__患者ID__2",
        "病程__治疗__方案",
    )


def test_read_only_profile_quarantines_missing_and_duplicate_ids(tmp_path):
    source = tmp_path / "synthetic.xlsx"
    _synthetic_workbook(source, ["SYN-001", "SYN-002", "SYN-002", None])
    before_hash = _sha256(source)
    before_files = set(tmp_path.iterdir())

    profile = profile_workbook(source)

    assert profile.row_count == 4
    assert profile.distinct_patient_count == 1
    assert profile.quarantine.missing_identifier_rows == 1
    assert profile.quarantine.duplicate_identifier_rows == 2
    assert profile.quarantine.duplicate_identifier_groups == 1
    assert profile.sheet_profiles[0].identifier_header.endswith("患者ID")
    assert _sha256(source) == before_hash
    assert set(tmp_path.iterdir()) == before_files

    # Neither dataclass representation nor the serialisable report leaks IDs.
    serialised = repr(profile) + repr(profile_to_dict(profile_workbooks([source])))
    assert "SYN-001" not in serialised
    assert "SYN-002" not in serialised


def test_cross_workbook_patient_set_checks_are_aggregate_only(tmp_path):
    left = tmp_path / "left.xlsx"
    right = tmp_path / "right.xlsx"
    _synthetic_workbook(left, ["SYN-001", "SYN-002"])
    _synthetic_workbook(right, ["SYN-002", "SYN-003"])

    profile = profile_workbooks([left, right])

    assert profile.total_rows == 4
    assert profile.summed_distinct_patient_count == 4
    assert profile.union_patient_count == 3
    assert profile.patients_in_all_workbooks == 1
    check = profile.patient_set_checks[0]
    assert check.intersection_count == 1
    assert check.union_count == 3
    assert check.left_only_count == check.right_only_count == 1
    report = repr(profile_to_dict(profile))
    for raw_identifier in ("SYN-001", "SYN-002", "SYN-003"):
        assert raw_identifier not in report


def test_profile_rejects_non_xlsx_input(tmp_path):
    source = tmp_path / "synthetic.csv"
    source.write_text("patient_id\nSYN-001\n", encoding="utf-8")

    with pytest.raises(WorkbookProfileError, match="only XLSX"):
        profile_workbook(source)
