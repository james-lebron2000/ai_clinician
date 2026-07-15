"""Local governance identity directory with role-bound independent credentials."""

from __future__ import annotations

import hmac
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence

from .privacy import is_placeholder_secret


class GovernanceAuthenticationError(PermissionError):
    """Raised when a governance credential or role cannot be verified."""


@dataclass(frozen=True, slots=True)
class GovernanceIdentity:
    subject: str
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{2,127}", self.subject):
            raise ValueError("governance subject is required")
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("governance identity requires unique roles")
        allowed = {
            "crc_clinical_expert",
            "methodologist",
            "data_steward",
            "safety_reviewer",
            "patient_representative",
            "research_governance",
            "expert_committee",
        }
        if not set(self.roles).issubset(allowed):
            raise ValueError("governance identity contains an unsupported role")


class GovernanceDirectory:
    """Maps opaque local tokens to provisioned subjects and roles."""

    def __init__(self, credentials: Mapping[str, Mapping[str, Any]]) -> None:
        entries: list[tuple[str, GovernanceIdentity]] = []
        for token, value in credentials.items():
            if len(token.encode("utf-8")) < 16:
                raise ValueError("governance tokens must contain at least 16 bytes")
            if is_placeholder_secret(token):
                raise ValueError("governance tokens cannot be placeholders")
            roles = value.get("roles", ())
            if isinstance(roles, str):
                roles = (roles,)
            identity = GovernanceIdentity(
                subject=str(value.get("subject", "")),
                roles=tuple(str(role) for role in roles),
            )
            entries.append((str(token), identity))
        self._entries = tuple(entries)

    @classmethod
    def from_env(cls) -> "GovernanceDirectory":
        encoded = os.environ.get("AI_CLINICIAN_GOVERNANCE_IDENTITIES")
        if not encoded:
            raise GovernanceAuthenticationError(
                "AI_CLINICIAN_GOVERNANCE_IDENTITIES is required for governed actions"
            )
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise GovernanceAuthenticationError(
                "governance identity directory is invalid"
            ) from exc
        if not isinstance(payload, Mapping):
            raise GovernanceAuthenticationError(
                "governance identity directory must be an object"
            )
        return cls(payload)

    def authenticate(
        self,
        token: str | None,
        *,
        allowed_roles: Sequence[str] | None = None,
    ) -> GovernanceIdentity:
        if token is None:
            raise GovernanceAuthenticationError("governance credential is required")
        matched: GovernanceIdentity | None = None
        for expected, identity in self._entries:
            if hmac.compare_digest(token, expected):
                matched = identity
        if matched is None:
            raise GovernanceAuthenticationError("invalid governance credential")
        if allowed_roles and not set(matched.roles) & set(allowed_roles):
            raise GovernanceAuthenticationError(
                "governance identity lacks the required role"
            )
        return matched


__all__ = [
    "GovernanceAuthenticationError",
    "GovernanceDirectory",
    "GovernanceIdentity",
]
