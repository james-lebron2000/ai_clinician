from __future__ import annotations

import subprocess

from ai_clinician.repository_safety import scan


def _git(tmp_path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)


def test_repository_safety_rejects_tracked_clinical_workbook(tmp_path):
    _git(tmp_path, "init")
    workbook = tmp_path / "patient_export.xlsx"
    workbook.write_bytes(b"synthetic-not-an-excel-file")
    _git(tmp_path, "add", "-f", workbook.name)

    findings = scan(tmp_path)

    assert any("forbidden file type" in finding for finding in findings)


def test_repository_safety_rejects_database_sidecars_and_xlsb(tmp_path):
    _git(tmp_path, "init")
    names = (
        "patient.db-wal",
        "patient.db-shm",
        "patient.db-journal",
        "patient.sqlite3-wal",
        "patient.xlsb",
    )
    for name in names:
        (tmp_path / name).write_bytes(b"synthetic-forbidden-artifact")
        _git(tmp_path, "add", "-f", name)

    findings = scan(tmp_path)

    for name in names:
        assert any(name in finding for finding in findings)


def test_repository_safety_accepts_source_code(tmp_path):
    _git(tmp_path, "init")
    source = tmp_path / "safe.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", source.name)

    assert scan(tmp_path) == []


def test_repository_safety_rejects_phi_in_tracked_text(tmp_path):
    _git(tmp_path, "init")
    note = tmp_path / "notes.txt"
    note.write_text("住院号: " + "123" * 3 + "\n", encoding="utf-8")
    _git(tmp_path, "add", note.name)

    findings = scan(tmp_path)

    assert any("possible PHI" in finding for finding in findings)


def test_repository_safety_scans_env_example_for_secrets(tmp_path):
    _git(tmp_path, "init")
    environment = tmp_path / ".env.example"
    token = "ghp_" + "A" * 30
    environment.write_text(f"GITHUB_TOKEN={token}\n", encoding="utf-8")
    _git(tmp_path, "add", environment.name)

    assert any("GitHub token" in item for item in scan(tmp_path))


def test_repository_safety_rejects_chinese_case_narrative(tmp_path):
    _git(tmp_path, "init")
    note = tmp_path / "summary.txt"
    note.write_text(
        "患者" + "王小明" + "于2024年接受治疗后进展\n", encoding="utf-8"
    )
    _git(tmp_path, "add", note.name)

    assert any("Chinese patient case narrative" in item for item in scan(tmp_path))
