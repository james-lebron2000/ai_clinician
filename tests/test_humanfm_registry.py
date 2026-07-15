from __future__ import annotations

from datetime import datetime, timezone

import pytest

from virtualhuman_agents.foundation.registry import HumanDataRegistry
from virtualhuman_agents.foundation.schemas import (
    AccessTier,
    AlignmentEdge,
    AlignmentEvidence,
    AllowedUse,
    BiologicalScale,
    ConsentMode,
    DatasetManifest,
    Modality,
    PairingLevel,
    SampleIndexRecord,
    SplitName,
)


def _manifest(pairing_level: PairingLevel) -> DatasetManifest:
    return DatasetManifest(
        id=f"dataset-{pairing_level.value}",
        version="1",
        title="Governed paired tissue dataset",
        source_name="test-site",
        source_uri="dms://test/dataset",
        data_controller="test-controller",
        license_or_dua="DUA-001",
        access_tier=AccessTier.CONTROLLED,
        modalities={Modality.HISTOPATHOLOGY, Modality.TRANSCRIPTOME},
        biological_scales={BiologicalScale.TISSUE},
        modality_scales={
            Modality.HISTOPATHOLOGY: BiologicalScale.TISSUE,
            Modality.TRANSCRIPTOME: BiologicalScale.TISSUE,
        },
        pairing_level=pairing_level,
        subject_key_namespace="site-test",
        subject_count=2,
        specimen_count=2,
        observation_count=4,
        site_ids=["SITE-A", "SITE-B"],
        geography=["CN"],
        allowed_uses={
            AllowedUse.AI_TRAINING,
            AllowedUse.AI_VALIDATION,
            AllowedUse.COMMERCIAL_RND,
        },
        permitted_compute_regions=["CN"],
        deidentification_method="project pseudonym",
        file_formats=["OME-TIFF", "Parquet"],
        checksum_manifest_uri="dms://test/checksums",
    )


def _asset(
    dataset_id: str,
    asset_id: str,
    modality: Modality,
    subject: str,
    specimen: str,
    split: SplitName,
    site: str,
) -> SampleIndexRecord:
    return SampleIndexRecord(
        id=asset_id,
        dataset_id=dataset_id,
        subject_key=subject,
        specimen_key=specimen,
        pairing_key=specimen,
        group_key=subject,
        modality=modality,
        biological_scale=BiologicalScale.TISSUE,
        site_id=site,
        acquisition_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        split=split,
        uri=f"dms://test/{asset_id}",
        sha256=(asset_id[0] * 64),
    )


def test_registry_detects_subject_leakage():
    registry = HumanDataRegistry()
    manifest = _manifest(PairingLevel.SAME_SPECIMEN)
    registry.register_dataset(manifest)
    registry.register_asset(
        _asset(
            manifest.id,
            "aaaaaaaa",
            Modality.HISTOPATHOLOGY,
            "subject-1",
            "specimen-1",
            SplitName.TRAIN,
            "SITE-A",
        )
    )
    registry.register_asset(
        _asset(
            manifest.id,
            "bbbbbbbb",
            Modality.TRANSCRIPTOME,
            "subject-1",
            "specimen-2",
            SplitName.INTERNAL_TEST,
            "SITE-A",
        )
    )
    report = registry.audit(compute_region="CN", commercial_project=True)
    assert not report.passed
    codes = {finding.code for finding in report.findings}
    assert "SUBJECT_SPLIT_LEAKAGE" in codes
    assert "GROUP_SPLIT_LEAKAGE" in codes


def test_pairing_level_prevents_fabricated_multimodal_pairs():
    registry = HumanDataRegistry()
    manifest = _manifest(PairingLevel.COHORT_UNPAIRED)
    registry.register_dataset(manifest)
    for asset in [
        _asset(
            manifest.id,
            "cccccccc",
            Modality.HISTOPATHOLOGY,
            "subject-1",
            "specimen-1",
            SplitName.TRAIN,
            "SITE-A",
        ),
        _asset(
            manifest.id,
            "dddddddd",
            Modality.TRANSCRIPTOME,
            "subject-1",
            "specimen-1",
            SplitName.TRAIN,
            "SITE-A",
        ),
    ]:
        registry.register_asset(asset)
    assert registry.alignment_pairs(Modality.HISTOPATHOLOGY, Modality.TRANSCRIPTOME) == []


def test_same_specimen_pair_is_available_for_alignment():
    registry = HumanDataRegistry()
    manifest = _manifest(PairingLevel.SAME_SPECIMEN)
    registry.register_dataset(manifest)
    for asset in [
        _asset(
            manifest.id,
            "eeeeeeee",
            Modality.HISTOPATHOLOGY,
            "subject-1",
            "specimen-1",
            SplitName.TRAIN,
            "SITE-A",
        ),
        _asset(
            manifest.id,
            "ffffffff",
            Modality.TRANSCRIPTOME,
            "subject-1",
            "specimen-1",
            SplitName.TRAIN,
            "SITE-A",
        ),
    ]:
        registry.register_asset(asset)
    pairs = registry.alignment_pairs(Modality.HISTOPATHOLOGY, Modality.TRANSCRIPTOME)
    assert len(pairs) == 1


def test_same_subject_temporal_link_is_not_a_strong_contrastive_pair():
    registry = HumanDataRegistry()
    manifest = _manifest(PairingLevel.SAME_SUBJECT)
    registry.register_dataset(manifest)
    for asset in [
        _asset(
            manifest.id,
            "12121212",
            Modality.HISTOPATHOLOGY,
            "subject-1",
            "specimen-1",
            SplitName.TRAIN,
            "SITE-A",
        ),
        _asset(
            manifest.id,
            "34343434",
            Modality.TRANSCRIPTOME,
            "subject-1",
            "specimen-2",
            SplitName.TRAIN,
            "SITE-A",
        ),
    ]:
        registry.register_asset(asset)

    assert registry.alignment_pairs(
        Modality.HISTOPATHOLOGY,
        Modality.TRANSCRIPTOME,
    ) == []
    assert len(
        registry.alignment_pairs(
            Modality.HISTOPATHOLOGY,
            Modality.TRANSCRIPTOME,
            strong_contrastive_only=False,
        )
    ) == 1


def test_individual_consent_mode_fails_closed_without_directive():
    registry = HumanDataRegistry()
    manifest = _manifest(PairingLevel.SAME_SPECIMEN).model_copy(
        update={"consent_mode": ConsentMode.INDIVIDUAL_DIRECTIVE}
    )
    registry.register_dataset(manifest)
    registry.register_asset(
        _asset(
            manifest.id,
            "abababab",
            Modality.HISTOPATHOLOGY,
            "subject-1",
            "specimen-1",
            SplitName.TRAIN,
            "SITE-A",
        )
    )
    report = registry.audit(compute_region="CN")
    assert "INDIVIDUAL_CONSENT" in {finding.code for finding in report.findings}


def test_alignment_edge_cannot_claim_an_exact_pair_for_different_specimens():
    registry = HumanDataRegistry()
    manifest = _manifest(PairingLevel.SAME_SUBJECT)
    registry.register_dataset(manifest)
    left = _asset(
        manifest.id,
        "56565656",
        Modality.HISTOPATHOLOGY,
        "subject-1",
        "specimen-1",
        SplitName.TRAIN,
        "SITE-A",
    )
    right = _asset(
        manifest.id,
        "78787878",
        Modality.TRANSCRIPTOME,
        "subject-1",
        "specimen-2",
        SplitName.TRAIN,
        "SITE-A",
    )
    registry.register_asset(left)
    registry.register_asset(right)

    with pytest.raises(ValueError, match="conflicts"):
        registry.register_alignment_edge(
            AlignmentEdge(
                dataset_id=manifest.id,
                left_asset_id=left.id,
                right_asset_id=right.id,
                pairing_level=PairingLevel.SAME_SPECIMEN,
                evidence=AlignmentEvidence.SOURCE_IDENTIFIER,
                confidence=1.0,
                evidence_sha256="9" * 64,
            )
        )


def test_locked_pairs_are_excluded_from_training_alignment_by_default():
    registry = HumanDataRegistry()
    manifest = _manifest(PairingLevel.SAME_SPECIMEN)
    registry.register_dataset(manifest)
    for asset in [
        _asset(
            manifest.id,
            "90909090",
            Modality.HISTOPATHOLOGY,
            "subject-2",
            "specimen-2",
            SplitName.INTERNAL_TEST,
            "SITE-B",
        ),
        _asset(
            manifest.id,
            "89898989",
            Modality.TRANSCRIPTOME,
            "subject-2",
            "specimen-2",
            SplitName.INTERNAL_TEST,
            "SITE-B",
        ),
    ]:
        registry.register_asset(asset)

    assert registry.alignment_pairs(
        Modality.HISTOPATHOLOGY,
        Modality.TRANSCRIPTOME,
    ) == []
    assert len(
        registry.alignment_pairs(
            Modality.HISTOPATHOLOGY,
            Modality.TRANSCRIPTOME,
            splits={SplitName.INTERNAL_TEST},
        )
    ) == 1
