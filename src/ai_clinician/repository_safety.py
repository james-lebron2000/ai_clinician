"""Repository policy that prevents clinical data and secrets entering Git."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".xls",
    ".csv",
    ".tsv",
    ".jsonl",
    ".parquet",
    ".feather",
    ".arrow",
    ".dcm",
    ".nii",
    ".vcf",
    ".cram",
    ".bam",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
    ".h5",
    ".pptx",
    ".docx",
    ".pkl",
    ".joblib",
}
FORBIDDEN_FILENAME_PATTERNS = {
    "database or sidecar": re.compile(
        r"(?i)\.(?:db|sqlite|sqlite3)(?:$|[-.])"
    ),
}
FORBIDDEN_PARTS = {
    "data",
    "derived",
    "raw",
    "quarantine",
    "patient_data",
    "outputs",
    "models",
    "checkpoints",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}
PHI_PATTERNS = {
    "Chinese national identifier": re.compile(
        r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"
    ),
    "mainland China mobile number": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "literal medical record identifier": re.compile(
        r"(?i)(?:patient[_ -]?id|mrn|medical[_ -]?record[_ -]?number|住院号|病案号)"
        r"\s*[:=]\s*['\"]?\d{6,}"
    ),
    "email address": re.compile(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    ),
    "labelled date of birth": re.compile(
        r"(?i)(?:date[_ -]?of[_ -]?birth|dob|出生日期)\s*[:=]\s*['\"]?"
        r"(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])"
    ),
    "labelled patient name": re.compile(
        r"(?:患者姓名|姓名)\s*[:=]\s*['\"]?[\u4e00-\u9fff]{2,4}"
    ),
    "Chinese patient case narrative": re.compile(
        r"(?:患者|病人)[：:\s]*[\u4e00-\u9fff]{2,4}(?:于|在)"
        r"(?:19|20)\d{2}年"
    ),
    "alphanumeric medical record identifier": re.compile(
        r"(?i)(?:mrn|medical[_ -]?record[_ -]?number|住院号|病案号)"
        r"\s*[:=]\s*['\"]?[A-Z0-9-]{5,}"
    ),
}
MAX_TRACKED_BYTES = 5 * 1024 * 1024


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    return [root / item.decode() for item in result.stdout.split(b"\0") if item]


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in tracked_files(root):
        relative = path.relative_to(root)
        lower_parts = {part.lower() for part in relative.parts[:-1]}
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if lower_parts & FORBIDDEN_PARTS:
            findings.append(f"forbidden path: {relative}")
        if suffixes & FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden file type: {relative}")
        for label, pattern in FORBIDDEN_FILENAME_PATTERNS.items():
            if pattern.search(path.name):
                findings.append(f"forbidden {label}: {relative}")
        if path.exists() and path.stat().st_size > MAX_TRACKED_BYTES:
            findings.append(f"tracked file exceeds 5 MiB: {relative}")
            continue
        if not path.exists():
            continue
        content = path.read_bytes()
        if b"\0" in content:
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"possible {label}: {relative}")
        for label, pattern in PHI_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"possible PHI ({label}): {relative}")
    return findings
