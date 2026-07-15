from __future__ import annotations

from typing import Any

from .architecture import HumanFoundationModel, PerturbationWorldModel
from .curriculum import build_default_curriculum
from .schemas import BiologicalScale, HumanFMConfig, Modality


def reference_humanfm_config() -> HumanFMConfig:
    return HumanFMConfig(
        name="HumanFM-reference",
        version="0.1.0",
        hidden_dim=128,
        num_attention_heads=4,
        modality_encoder_layers=1,
        cross_scale_layers=1,
        scale_latent_tokens=2,
        maximum_tokens_per_modality=64,
        state_dim=64,
        intervention_dim=16,
        modality_input_dims={
            Modality.GENOME: 32,
            Modality.SINGLE_CELL: 64,
            Modality.HISTOPATHOLOGY: 48,
            Modality.RADIOLOGY: 40,
            Modality.LABORATORY: 16,
            Modality.WEARABLE: 8,
        },
        modality_scales={
            Modality.GENOME: BiologicalScale.MOLECULE,
            Modality.SINGLE_CELL: BiologicalScale.CELL,
            Modality.HISTOPATHOLOGY: BiologicalScale.TISSUE,
            Modality.RADIOLOGY: BiologicalScale.ORGAN,
            Modality.LABORATORY: BiologicalScale.PERSON,
            Modality.WEARABLE: BiologicalScale.PERSON,
        },
        context_of_use=(
            "Research-only cross-scale representation and perturbation hypothesis generation; "
            "not a stand-alone clinical decision."
        ),
    )


def run_humanfm_demo(seed: int = 17) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    config = reference_humanfm_config()
    model = HumanFoundationModel(config)
    world_model = PerturbationWorldModel(config)
    batch_size = 3
    features = {
        Modality.GENOME: torch.randn(batch_size, 4, 32),
        Modality.SINGLE_CELL: torch.randn(batch_size, 12, 64),
        Modality.HISTOPATHOLOGY: torch.randn(batch_size, 16, 48),
        Modality.LABORATORY: torch.randn(batch_size, 6, 16),
    }
    masks = {
        modality: torch.ones(values.shape[:2], dtype=torch.bool)
        for modality, values in features.items()
    }
    # Demonstrate missing cells/tiles for one subject without fabricating values.
    masks[Modality.SINGLE_CELL][2, 8:] = False
    output = model(features, masks, time_delta=torch.tensor([0.0, 7.0, 30.0]))
    transition = world_model(
        output.state,
        torch.randn(batch_size, config.intervention_dim),
        torch.tensor([1.0, 7.0, 14.0]),
    )
    curriculum = build_default_curriculum(config)
    return {
        "model": config.name,
        "version": config.version,
        "parameter_count": model.parameter_count(),
        "state_shape": list(output.state.shape),
        "next_state_shape": list(transition.next_state_mean.shape),
        "available_modalities": [item.value for item in output.available_modalities],
        "scale_embeddings": {
            scale.value: list(value.shape) for scale, value in output.scale_embeddings.items()
        },
        "training_stages": [stage.name for stage in curriculum],
        "locked_final_stage": curriculum[-1].locked,
        "notice": "Synthetic tensor smoke test only; no biological or clinical claim.",
    }
