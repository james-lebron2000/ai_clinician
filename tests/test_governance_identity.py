from __future__ import annotations

import pytest

from ai_clinician.identity import GovernanceAuthenticationError, GovernanceDirectory
from ai_clinician.store import ResearchStore, _OBJECT_SCHEMAS


def test_governance_directory_binds_subject_and_role_to_credential() -> None:
    directory = GovernanceDirectory(
        {
            "clinical-reviewer-token": {
                "subject": "crc-reviewer-1",
                "roles": ["crc_clinical_expert"],
            }
        }
    )
    identity = directory.authenticate(
        "clinical-reviewer-token", allowed_roles=["crc_clinical_expert"]
    )
    assert identity.subject == "crc-reviewer-1"
    with pytest.raises(GovernanceAuthenticationError, match="required role"):
        directory.authenticate(
            "clinical-reviewer-token", allowed_roles=["methodologist"]
        )


def test_governance_directory_rejects_documentation_placeholder() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        GovernanceDirectory(
            {
                "replace-with-an-independently-provisioned-token": {
                    "subject": "crc-reviewer-1",
                    "roles": ["crc_clinical_expert"],
                }
            }
        )


def test_every_governed_schema_has_an_explicit_write_role() -> None:
    synthetic_payload = {
        "role": "crc_clinical_expert",
        "channel": "ai_candidate",
        "status": "draft",
    }
    missing = {
        object_type
        for object_type in _OBJECT_SCHEMAS
        if not ResearchStore._required_write_roles(object_type, synthetic_payload)
    }
    assert missing == set()
