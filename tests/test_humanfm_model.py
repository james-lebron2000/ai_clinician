from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from virtualhuman_agents.foundation.architecture import (  # noqa: E402
    HumanFoundationModel,
    PerturbationWorldModel,
    contrastive_alignment_loss,
    gaussian_transition_loss,
)
from virtualhuman_agents.foundation.curriculum import build_default_curriculum  # noqa: E402
from virtualhuman_agents.foundation.demo import reference_humanfm_config  # noqa: E402
from virtualhuman_agents.foundation.model_registry import FoundationModelRegistry  # noqa: E402
from virtualhuman_agents.foundation.schemas import (  # noqa: E402
    DatasetSnapshot,
    FoundationReadinessReport,
    ModelRelease,
    Modality,
)


def test_cross_scale_model_supports_missing_modalities():
    config = reference_humanfm_config()
    model = HumanFoundationModel(config)
    features = {
        Modality.GENOME: torch.randn(2, 3, 32),
        Modality.HISTOPATHOLOGY: torch.randn(2, 5, 48),
        Modality.LABORATORY: torch.randn(2, 4, 16),
    }
    masks = {
        modality: torch.ones(values.shape[:2], dtype=torch.bool)
        for modality, values in features.items()
    }
    masks[Modality.HISTOPATHOLOGY][1] = False
    output = model(features, masks, time_delta=torch.tensor([0.0, 30.0]))
    assert output.state.shape == (2, config.state_dim)
    assert torch.isfinite(output.state).all()
    assert set(output.available_modalities) == set(features)
    assert len(output.scale_embeddings) == len(config.active_scales)
    assert model.parameter_count() > 0


def test_model_rejects_mask_for_an_absent_modality():
    config = reference_humanfm_config()
    model = HumanFoundationModel(config)
    with pytest.raises(ValueError, match="absent modalities"):
        model(
            {Modality.GENOME: torch.randn(2, 3, 32)},
            {Modality.RADIOLOGY: torch.ones(2, 3, dtype=torch.bool)},
        )


def test_perturbation_world_model_and_losses():
    config = reference_humanfm_config()
    world_model = PerturbationWorldModel(config)
    state = torch.randn(3, config.state_dim)
    intervention = torch.randn(3, config.intervention_dim)
    output = world_model(state, intervention, torch.tensor([1.0, 2.0, 3.0]))
    assert output.next_state_mean.shape == state.shape
    loss = gaussian_transition_loss(output, torch.randn_like(state))
    assert torch.isfinite(loss)
    alignment = contrastive_alignment_loss(torch.randn(3, 8), torch.randn(3, 8))
    assert torch.isfinite(alignment)


def test_curriculum_ends_in_non_trainable_locked_validation():
    curriculum = build_default_curriculum(reference_humanfm_config())
    assert curriculum[-1].locked is True
    assert curriculum[-1].trainable_components == ["none"]


def test_agent_can_only_resolve_released_exact_cou_model():
    registry = FoundationModelRegistry()
    snapshot = DatasetSnapshot(
        manifest_ids=["dataset-1"],
        asset_ids=["asset-1"],
        split_manifest_sha256="1" * 64,
        consent_decision_sha256="2" * 64,
        preprocessing_sha256="3" * 64,
        ontology_versions={"CL": "test"},
        created_by_user_id="data-steward",
        snapshot_sha256="4" * 64,
    )
    registry.register_snapshot(snapshot)
    cou = "Research-only cross-scale representation for a locked benchmark."
    release = ModelRelease(
        model_name="HumanFM-test",
        model_version="1",
        dataset_snapshot_ids=[snapshot.id],
        config_sha256="5" * 64,
        code_sha256="6" * 64,
        container_digest="sha256:" + "7" * 64,
        weights_sha256="8" * 64,
        preprocessing_sha256="9" * 64,
        context_of_use=cou,
        forbidden_uses=["stand-alone diagnosis"],
        ood_detector_release_id="ood-1",
    )
    registry.register_release(release)
    with pytest.raises(KeyError):
        registry.resolve_for_agent(release.model_name, cou)
    readiness = FoundationReadinessReport(
        model_name=release.model_name,
        model_version=release.model_version,
        context_of_use=cou,
        registry_audit_passed=True,
        passed=True,
    )
    registry.promote(release.id, readiness)
    assert registry.resolve_for_agent(release.model_name, cou).status == "released"
