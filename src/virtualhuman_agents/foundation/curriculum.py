from __future__ import annotations

from .schemas import (
    HumanFMConfig,
    Modality,
    PairingLevel,
    TrainingObjective,
    TrainingStageSpec,
)


def build_default_curriculum(config: HumanFMConfig) -> list[TrainingStageSpec]:
    modalities = set(config.modality_input_dims)
    temporal_modalities = modalities & {
        Modality.PHYSIOLOGICAL_WAVEFORM,
        Modality.WEARABLE,
        Modality.LABORATORY,
        Modality.MEDICATION,
        Modality.PERTURBATION,
        Modality.OUTCOME,
        Modality.MICROSCOPY,
    }
    stages = [
        TrainingStageSpec(
            index=1,
            name="unimodal_self_supervision",
            objectives=[
                TrainingObjective.MASKED_RECONSTRUCTION,
                TrainingObjective.SELF_DISTILLATION,
            ],
            required_modalities=set(),
            trainable_components=["modality_encoders", "unimodal_decoders"],
            exit_criteria={"minimum_linear_probe_gain": 0.02},
        ),
        TrainingStageSpec(
            index=2,
            name="governed_cross_modal_alignment",
            objectives=[TrainingObjective.CONTRASTIVE_ALIGNMENT],
            required_modalities=modalities,
            permitted_pairing_levels={
                PairingLevel.SAME_MEASUREMENT,
                PairingLevel.SAME_SPECIMEN,
            },
            trainable_components=["modality_encoders", "scale_attention"],
            exit_criteria={"minimum_retrieval_recall_at_10": 0.70},
        ),
        TrainingStageSpec(
            index=3,
            name="temporal_perturbation_world_model",
            objectives=[
                TrainingObjective.TEMPORAL_PREDICTION,
                TrainingObjective.PERTURBATION_RESPONSE,
            ],
            required_modalities=temporal_modalities,
            permitted_pairing_levels={
                PairingLevel.SAME_MEASUREMENT,
                PairingLevel.SAME_SPECIMEN,
                PairingLevel.SAME_SUBJECT_TIME_WINDOW,
                PairingLevel.SAME_SUBJECT,
            },
            trainable_components=["time_encoder", "perturbation_world_model"],
            frozen_components=["validated_unimodal_backbones"],
            exit_criteria={"minimum_baseline_mae_reduction": 0.10},
        ),
        TrainingStageSpec(
            index=4,
            name="cross_scale_state_learning",
            objectives=[TrainingObjective.CROSS_SCALE_CONSISTENCY],
            required_modalities=modalities,
            permitted_pairing_levels={
                PairingLevel.SAME_MEASUREMENT,
                PairingLevel.SAME_SPECIMEN,
                PairingLevel.SAME_SUBJECT_TIME_WINDOW,
                PairingLevel.SAME_SUBJECT,
            },
            trainable_components=["scale_attention", "cross_scale_encoder"],
            exit_criteria={"minimum_external_task_pass_rate": 0.80},
        ),
        TrainingStageSpec(
            index=5,
            name="context_specific_adaptation",
            objectives=[TrainingObjective.TASK_SUPERVISION],
            required_modalities=set(),
            trainable_components=["adapters", "task_heads", "calibration_heads"],
            frozen_components=["foundation_backbone"],
            exit_criteria={"minimum_context_of_use_gate_pass_rate": 1.0},
        ),
        TrainingStageSpec(
            index=6,
            name="locked_prospective_validation",
            objectives=[TrainingObjective.TASK_SUPERVISION],
            required_modalities=set(),
            trainable_components=["none"],
            frozen_components=["all_model_parameters", "preprocessing", "thresholds"],
            locked=True,
            exit_criteria={"minimum_critical_gate_pass_rate": 1.0},
        ),
    ]
    validate_curriculum(stages)
    return stages


def validate_curriculum(stages: list[TrainingStageSpec]) -> None:
    indices = [stage.index for stage in stages]
    if indices != list(range(1, len(stages) + 1)):
        raise ValueError("training stages must be sequential and start at one")
    locked_indices = [index for index, stage in enumerate(stages) if stage.locked]
    if locked_indices and locked_indices != [len(stages) - 1]:
        raise ValueError("only the final stage may be locked")
    if stages[-1].locked and stages[-1].trainable_components != ["none"]:
        raise ValueError("locked prospective validation cannot train model components")
