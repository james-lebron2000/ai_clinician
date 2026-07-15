"""Runtime privacy boundaries for approved, local-only clinical research.

The source workbooks are deliberately outside the repository.  This module
provides the only supported path resolution and pseudonymisation primitives;
it never returns or logs a source identifier.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PrivacyBoundaryError(ValueError):
    """Raised when a path or payload crosses the approved data boundary."""


_PLACEHOLDER_SECRET_MARKERS = (
    "replace-with",
    "provision-",
    "change-me",
    "changeme",
    "placeholder",
    "approved-local-api-key",
    "independently-provisioned-token",
)


def is_placeholder_secret(value: str | bytes) -> bool:
    """Recognise documentation placeholders that must never become credentials."""

    normalized = (
        value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
    ).strip().casefold()
    return any(marker in normalized for marker in _PLACEHOLDER_SECRET_MARKERS)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _containing_git_worktree(path: Path) -> Path | None:
    """Find a Git worktree by filesystem markers, including linked worktrees."""

    resolved = _resolved(path)
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _discover_git_root(start: Path) -> Path | None:
    candidate = start if start.is_dir() else start.parent
    result = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return _resolved(result.stdout.strip())
    return None


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    """Validated roots for read-only source data and patient-level derivatives."""

    raw_root: Path
    derived_root: Path
    git_root: Path

    def __post_init__(self) -> None:
        raw = _resolved(self.raw_root)
        derived = _resolved(self.derived_root)
        git = _resolved(self.git_root)
        object.__setattr__(self, "raw_root", raw)
        object.__setattr__(self, "derived_root", derived)
        object.__setattr__(self, "git_root", git)

        if not raw.is_dir():
            raise PrivacyBoundaryError(f"raw root is not an existing directory: {raw}")
        if not git.is_dir():
            raise PrivacyBoundaryError(f"Git work tree is not a directory: {git}")
        if _overlaps(raw, git):
            raise PrivacyBoundaryError("raw root must be outside the Git work tree")
        if _overlaps(derived, git):
            raise PrivacyBoundaryError("derived root must be outside the Git work tree")
        if _overlaps(raw, derived):
            raise PrivacyBoundaryError("raw and derived roots must not overlap")
        for label, root in (("raw", raw), ("derived", derived)):
            containing_worktree = _containing_git_worktree(root)
            if containing_worktree is not None:
                raise PrivacyBoundaryError(
                    f"{label} root must be outside every Git work tree"
                )

    @classmethod
    def from_env(cls, *, git_root: str | Path | None = None) -> "PrivacyConfig":
        raw = os.environ.get("AI_CLINICIAN_RAW_ROOT")
        derived = os.environ.get("AI_CLINICIAN_DERIVED_ROOT")
        if not raw or not derived:
            raise PrivacyBoundaryError(
                "AI_CLINICIAN_RAW_ROOT and AI_CLINICIAN_DERIVED_ROOT are required"
            )
        discovered_roots = [
            root
            for root in (
                _discover_git_root(Path.cwd()),
                _discover_git_root(Path(__file__).resolve()),
            )
            if root is not None
        ]
        discovered_root = discovered_roots[-1] if discovered_roots else None
        if git_root is not None:
            configured = _resolved(git_root)
            if discovered_root is not None and configured != discovered_root:
                raise PrivacyBoundaryError(
                    "configured Git root does not match the discovered work tree"
                )
            discovered_root = configured
        configured_git_root = discovered_root or Path(__file__).resolve().parents[2]
        return cls(
            raw_root=Path(raw),
            derived_root=Path(derived),
            git_root=Path(configured_git_root),
        )

    def resolve_raw(self, relative_path: str | Path, *, must_exist: bool = True) -> Path:
        """Resolve a source path without permitting traversal outside raw_root."""

        candidate = _resolved(self.raw_root / relative_path)
        if not candidate.is_relative_to(self.raw_root):
            raise PrivacyBoundaryError("raw path escapes AI_CLINICIAN_RAW_ROOT")
        if must_exist and not candidate.exists():
            raise PrivacyBoundaryError(f"raw source does not exist: {candidate.name}")
        return candidate

    def resolve_derived(
        self, relative_path: str | Path, *, create_parent: bool = False
    ) -> Path:
        """Resolve an approved output path without touching the Git work tree."""

        candidate = _resolved(self.derived_root / relative_path)
        if not candidate.is_relative_to(self.derived_root):
            raise PrivacyBoundaryError("derived path escapes AI_CLINICIAN_DERIVED_ROOT")
        if create_parent:
            candidate.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        return candidate


def pseudonymize_identifier(identifier: str, *, secret: str | bytes) -> str:
    """Return a stable HMAC token; the identifier and secret are never persisted."""

    cleaned = identifier.strip()
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    if not cleaned:
        raise PrivacyBoundaryError("cannot pseudonymize an empty identifier")
    if len(key) < 16:
        raise PrivacyBoundaryError("pseudonymization secret must contain at least 16 bytes")
    if is_placeholder_secret(secret):
        raise PrivacyBoundaryError("pseudonymization secret cannot be a placeholder")
    digest = hmac.new(key, cleaned.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"pt_{digest}"


# These fields are patient-level even when their values happen to be hashed.
_PATIENT_LEVEL_KEYS = {
    "patient_id",
    "patient_key",
    "patient_name",
    "subject_id",
    "person_id",
    "case_id",
    "encounter_id",
    "record_id",
    "medical_record_number",
    "mrn",
    "raw_text",
    "source_text",
    "clinical_note",
    "note",
    "notes",
    "note_text",
    "free_text",
    "evidence_excerpt",
    "excerpt",
    "cell_ref",
    "event_time",
    "date_of_birth",
    "birth_date",
    "dob",
    "address",
    "email",
    "phone",
    "住院号",
    "病案号",
    "患者姓名",
    "身份证号",
    "手机号",
    "病例号",
    "姓名",
    "出生日期",
    "家庭住址",
}

_PATIENT_VALUE_PATTERNS = (
    re.compile(
        r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"
    ),
    re.compile(r"(?<![A-Za-z0-9])1[3-9]\d{9}(?![A-Za-z0-9])"),
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?i)\b(?:MRN|病案号|住院号)[-_: ]*[A-Z0-9]{5,}\b"),
    re.compile(r"\bpt_[a-fA-F0-9]{64}\b"),
    re.compile(
        r"(?:患者|病人)[：:\s]*[\u4e00-\u9fff]{2,4}(?:于|在)"
        r"(?:19|20)\d{2}年"
    ),
)


def assert_aggregate_safe_payload(payload: Any, *, path: str = "$") -> None:
    """Reject patient-level fields before an object reaches the research store/API."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().casefold()
            if normalized in _PATIENT_LEVEL_KEYS:
                raise PrivacyBoundaryError(f"patient-level field is forbidden at {path}.{key}")
            assert_aggregate_safe_payload(value, path=f"{path}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        for index, value in enumerate(payload):
            assert_aggregate_safe_payload(value, path=f"{path}[{index}]")
        return
    if isinstance(payload, str) and any(
        pattern.search(payload) for pattern in _PATIENT_VALUE_PATTERNS
    ):
        raise PrivacyBoundaryError(f"possible patient identifier is forbidden at {path}")


__all__ = [
    "PrivacyBoundaryError",
    "PrivacyConfig",
    "assert_aggregate_safe_payload",
    "is_placeholder_secret",
    "pseudonymize_identifier",
]
