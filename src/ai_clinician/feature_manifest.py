"""Locked column-role manifest for longitudinal state discovery.

Column names are not semantically meaningful enough to prevent treatment
leakage (a treatment can be called ``x7``).  State fitting therefore accepts
only a complete, content-addressed manifest that assigns exactly one role to
every input column.  The digest is always recomputed from canonical content;
it is never treated as a self-standing approval token.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ManifestValidationError(ValueError):
    """Raised when analysis columns are not bound to a valid manifest."""


class ColumnRole(StrEnum):
    """Exhaustive roles permitted in the state-discovery input table."""

    PATIENT_ID = "patient_id"
    INDEX_DATE = "index_date"
    EVENT_TIME = "event_time"
    FEATURE_AVAILABLE_TIME = "feature_available_time"
    STATE_FEATURE = "state_feature"
    TREATMENT_ACTION = "treatment_action"
    OUTCOME = "outcome"
    CENSORING = "censoring"
    STRATIFIER = "stratifier"


class AnalysisColumn(BaseModel):
    """One reviewed input column and its single analysis role."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1)
    role: ColumnRole


class AnalysisColumnManifest(BaseModel):
    """Canonical, locked schema for all columns passed to state discovery."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    columns: tuple[AnalysisColumn, ...] = Field(min_length=1)
    locked: Literal[True] = True

    @model_validator(mode="after")
    def validate_roles(self) -> "AnalysisColumnManifest":
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("manifest column names must be unique")
        normalized = [_normalize_column_name(name) for name in names]
        if len(normalized) != len(set(normalized)):
            raise ValueError("manifest column names must be unique after normalization")

        for singleton in (
            ColumnRole.PATIENT_ID,
            ColumnRole.INDEX_DATE,
            ColumnRole.EVENT_TIME,
            ColumnRole.FEATURE_AVAILABLE_TIME,
        ):
            if len(self.names_for(singleton)) != 1:
                raise ValueError(f"manifest requires exactly one {singleton.value} column")
        if not self.names_for(ColumnRole.STATE_FEATURE):
            raise ValueError("manifest requires at least one state_feature column")
        if not self.names_for(ColumnRole.TREATMENT_ACTION):
            raise ValueError("manifest requires at least one treatment_action column")
        return self

    def names_for(self, role: ColumnRole) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.role is role)

    @property
    def role_by_name(self) -> dict[str, ColumnRole]:
        return {column.name: column.role for column in self.columns}

    def canonical_payload(self) -> dict[str, Any]:
        """Return deterministic content used for the manifest digest.

        Column order is intentionally preserved because it fixes feature
        matrix order and therefore is part of the fitted-model contract.
        """

        return {
            "columns": [
                {"name": column.name, "role": column.role.value}
                for column in self.columns
            ],
            "locked": True,
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def verify_digest(self, claimed_sha256: str) -> str:
        normalized = str(claimed_sha256).strip().casefold()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ManifestValidationError("analysis manifest SHA-256 is malformed")
        if normalized != self.sha256:
            raise ManifestValidationError(
                "analysis manifest SHA-256 does not match canonical manifest content"
            )
        return normalized

    def validate_analysis_input(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        feature_names: Sequence[str],
        treatment_columns: Sequence[str],
        patient_id_column: str,
        index_date_column: str,
        event_time_column: str,
        feature_available_time_column: str,
    ) -> None:
        """Bind runtime columns and model inputs to the reviewed manifest."""

        if not records:
            raise ManifestValidationError("at least one analysis record is required")

        expected_columns = set(self.role_by_name)
        for row_number, record in enumerate(records):
            if any(not isinstance(name, str) for name in record):
                raise ManifestValidationError(
                    f"analysis input column names must be strings at record {row_number}"
                )
            actual_columns = {str(name) for name in record}
            undeclared = actual_columns - expected_columns
            missing = expected_columns - actual_columns
            if undeclared:
                raise ManifestValidationError(
                    "analysis input contains columns absent from the locked manifest "
                    f"at record {row_number}: {', '.join(sorted(undeclared))}"
                )
            if missing:
                raise ManifestValidationError(
                    "analysis input is missing locked manifest columns "
                    f"at record {row_number}: {', '.join(sorted(missing))}"
                )

        declared_features = self.names_for(ColumnRole.STATE_FEATURE)
        requested_features = tuple(feature_names)
        if requested_features != declared_features:
            raise ManifestValidationError(
                "feature_names must exactly match state_feature columns in manifest order"
            )

        declared_treatments = self.names_for(ColumnRole.TREATMENT_ACTION)
        requested_treatments = tuple(treatment_columns)
        if requested_treatments != declared_treatments:
            raise ManifestValidationError(
                "treatment_columns must exactly match treatment_action columns in manifest order"
            )
        if set(requested_features) & set(requested_treatments):
            raise ManifestValidationError(
                "treatment_action columns cannot appear in feature_names"
            )

        required_bindings = {
            patient_id_column: ColumnRole.PATIENT_ID,
            index_date_column: ColumnRole.INDEX_DATE,
            event_time_column: ColumnRole.EVENT_TIME,
            feature_available_time_column: ColumnRole.FEATURE_AVAILABLE_TIME,
        }
        roles = self.role_by_name
        for column_name, expected_role in required_bindings.items():
            if roles.get(column_name) is not expected_role:
                raise ManifestValidationError(
                    f"column {column_name!r} must be classified as {expected_role.value}"
                )

        # An action column must be material, not a fictitious name used to
        # satisfy the API while an opaque action (for example x7) is fitted.
        for treatment_column in declared_treatments:
            if not all(treatment_column in record for record in records):
                raise ManifestValidationError(
                    f"treatment_action column {treatment_column!r} does not exist in every record"
                )
            if not any(
                record[treatment_column] is not None
                and (
                    not isinstance(record[treatment_column], str)
                    or record[treatment_column].strip()
                )
                for record in records
            ):
                raise ManifestValidationError(
                    f"treatment_action column {treatment_column!r} has no observed values"
                )


class AnalysisManifestRegistration(BaseModel):
    """Governed, cohort-bound registration of the semantic column manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: f"analysis_manifest_{uuid4().hex}", min_length=3
    )
    manifest: AnalysisColumnManifest
    cohort_snapshot_id: str = Field(min_length=3)
    cohort_snapshot_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    source_schema_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    registered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    locked: Literal[True] = True
    research_only: Literal[True] = True

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"registered_at"})
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()


def _normalize_column_name(name: str) -> str:
    return "_".join(name.casefold().strip().replace("-", "_").split())


def parse_analysis_manifest(value: Any) -> AnalysisColumnManifest:
    """Parse the public manifest wire shape without accepting loose schemas."""

    if isinstance(value, AnalysisColumnManifest):
        return value
    if value is None:
        raise ManifestValidationError("a locked analysis column manifest is required")
    try:
        return AnalysisColumnManifest.model_validate(value)
    except Exception as exc:
        raise ManifestValidationError("invalid analysis column manifest") from exc


def canonical_manifest_sha256(value: Any) -> str:
    """Convenience helper used when preparing a locked synthetic input file."""

    return parse_analysis_manifest(value).sha256


__all__ = [
    "AnalysisColumn",
    "AnalysisColumnManifest",
    "AnalysisManifestRegistration",
    "ColumnRole",
    "ManifestValidationError",
    "canonical_manifest_sha256",
    "parse_analysis_manifest",
]
