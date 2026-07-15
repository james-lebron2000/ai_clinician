from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import BiologicalScale, HumanFMConfig, Modality

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without foundation extra
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class HumanFMOutput:
    state: Tensor
    scale_embeddings: dict[BiologicalScale, Tensor]
    modality_embeddings: dict[Modality, Tensor]
    available_modalities: tuple[Modality, ...]


@dataclass
class TransitionOutput:
    next_state_mean: Tensor
    next_state_log_variance: Tensor


if nn is not None:

    class ModalityTokenEncoder(nn.Module):
        def __init__(self, input_dim: int, config: HumanFMConfig) -> None:
            super().__init__()
            self.maximum_tokens = config.maximum_tokens_per_modality
            self.projection = nn.Linear(input_dim, config.hidden_dim)
            self.position = nn.Parameter(
                torch.zeros(1, self.maximum_tokens, config.hidden_dim)
            )
            self.modality_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=config.modality_encoder_layers,
                enable_nested_tensor=False,
            )
            self.normalization = nn.LayerNorm(config.hidden_dim)
            nn.init.trunc_normal_(self.position, std=0.02)
            nn.init.trunc_normal_(self.modality_token, std=0.02)

        def forward(self, values: Tensor, valid_mask: Tensor | None = None) -> Tensor:
            if values.ndim == 2:
                values = values.unsqueeze(1)
            if values.ndim != 3:
                raise ValueError("modality features must have shape [batch, tokens, features]")
            if values.shape[1] > self.maximum_tokens:
                raise ValueError(
                    f"token count {values.shape[1]} exceeds configured maximum {self.maximum_tokens}"
                )
            if valid_mask is not None and valid_mask.shape != values.shape[:2]:
                raise ValueError("valid mask must have shape [batch, tokens]")
            if valid_mask is not None:
                valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)
            hidden = self.projection(values)
            hidden = hidden + self.position[:, : hidden.shape[1]] + self.modality_token
            padding_mask = None if valid_mask is None else ~valid_mask.bool()
            hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
            return self.normalization(hidden)


    class HumanFoundationModel(nn.Module):
        """Reference cross-scale fusion core.

        Raw modalities are converted to token features by governed adapters. This
        module fuses those tokens without pretending that unpaired cohorts are
        individual-level multimodal observations.
        """

        def __init__(self, config: HumanFMConfig) -> None:
            super().__init__()
            self.config = config
            self.modality_encoders = nn.ModuleDict(
                {
                    modality.value: ModalityTokenEncoder(input_dim, config)
                    for modality, input_dim in config.modality_input_dims.items()
                }
            )
            self.scale_latents = nn.ParameterDict(
                {
                    scale.value: nn.Parameter(
                        torch.zeros(
                            1,
                            config.scale_latent_tokens,
                            config.hidden_dim,
                        )
                    )
                    for scale in config.active_scales
                }
            )
            self.scale_positions = nn.Parameter(
                torch.zeros(1, len(config.active_scales), config.hidden_dim)
            )
            self.missing_scale_tokens = nn.ParameterDict(
                {
                    scale.value: nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
                    for scale in config.active_scales
                }
            )
            self.scale_attention = nn.ModuleDict(
                {
                    scale.value: nn.MultiheadAttention(
                        config.hidden_dim,
                        config.num_attention_heads,
                        dropout=config.dropout,
                        batch_first=True,
                    )
                    for scale in config.active_scales
                }
            )
            cross_scale_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cross_scale_encoder = nn.TransformerEncoder(
                cross_scale_layer,
                num_layers=config.cross_scale_layers,
                enable_nested_tensor=False,
            )
            self.time_encoder = nn.Sequential(
                nn.Linear(1, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.state_projection = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.state_dim),
            )
            self._reset_parameters()

        def _reset_parameters(self) -> None:
            for value in self.scale_latents.values():
                nn.init.trunc_normal_(value, std=0.02)
            for value in self.missing_scale_tokens.values():
                nn.init.trunc_normal_(value, std=0.02)
            nn.init.trunc_normal_(self.scale_positions, std=0.02)

        def forward(
            self,
            modality_features: dict[Modality | str, Tensor],
            modality_masks: dict[Modality | str, Tensor] | None = None,
            time_delta: Tensor | None = None,
        ) -> HumanFMOutput:
            if not modality_features:
                raise ValueError("at least one modality is required")
            normalized_features = {Modality(key): value for key, value in modality_features.items()}
            unknown = set(normalized_features) - set(self.config.modality_input_dims)
            if unknown:
                raise ValueError(f"unconfigured modalities: {sorted(item.value for item in unknown)}")
            normalized_masks = {
                Modality(key): value for key, value in (modality_masks or {}).items()
            }
            orphan_masks = set(normalized_masks) - set(normalized_features)
            if orphan_masks:
                raise ValueError(
                    "masks were supplied for absent modalities: "
                    f"{sorted(item.value for item in orphan_masks)}"
                )
            first = next(iter(normalized_features.values()))
            batch_size = first.shape[0]
            if any(values.shape[0] != batch_size for values in normalized_features.values()):
                raise ValueError("all modalities must have the same batch size")
            if any(values.device != first.device for values in normalized_features.values()):
                raise ValueError("all modality tensors must be on the same device")

            encoded: dict[Modality, Tensor] = {}
            masks: dict[Modality, Tensor] = {}
            for modality, values in normalized_features.items():
                if values.ndim == 2:
                    token_count = 1
                elif values.ndim == 3:
                    token_count = values.shape[1]
                else:
                    raise ValueError("modality features must be rank 2 or 3")
                mask = normalized_masks.get(modality)
                if mask is None:
                    mask = torch.ones(
                        batch_size, token_count, dtype=torch.bool, device=values.device
                    )
                else:
                    mask = mask.to(device=values.device, dtype=torch.bool)
                encoded[modality] = self.modality_encoders[modality.value](values, mask)
                masks[modality] = mask

            scale_vectors: list[Tensor] = []
            scale_embeddings: dict[BiologicalScale, Tensor] = {}
            for scale_index, scale in enumerate(self.config.active_scales):
                query = self.scale_latents[scale.value].expand(batch_size, -1, -1)
                routed = [
                    modality
                    for modality in encoded
                    if self.config.modality_scales[modality] is scale
                ]
                if routed:
                    keys = torch.cat([encoded[item] for item in routed], dim=1)
                    valid = torch.cat([masks[item] for item in routed], dim=1)
                    missing = self.missing_scale_tokens[scale.value].expand(batch_size, -1, -1)
                    keys = torch.cat([keys, missing], dim=1)
                    valid = torch.cat(
                        [
                            valid,
                            torch.ones(
                                batch_size, 1, dtype=torch.bool, device=valid.device
                            ),
                        ],
                        dim=1,
                    )
                    attended, _ = self.scale_attention[scale.value](
                        query,
                        keys,
                        keys,
                        key_padding_mask=~valid,
                        need_weights=False,
                    )
                    scale_hidden = query + attended
                else:
                    scale_hidden = query + self.missing_scale_tokens[scale.value]
                pooled = scale_hidden.mean(dim=1)
                if scale is BiologicalScale.PERSON and time_delta is not None:
                    if time_delta.ndim == 1:
                        time_delta = time_delta.unsqueeze(-1)
                    if time_delta.shape != (batch_size, 1):
                        raise ValueError("time_delta must have shape [batch] or [batch, 1]")
                    pooled = pooled + self.time_encoder(
                        time_delta.to(device=pooled.device, dtype=pooled.dtype)
                    )
                pooled = pooled + self.scale_positions[:, scale_index]
                scale_vectors.append(pooled)
                scale_embeddings[scale] = pooled

            cross_scale = self.cross_scale_encoder(torch.stack(scale_vectors, dim=1))
            for index, scale in enumerate(self.config.active_scales):
                scale_embeddings[scale] = cross_scale[:, index]
            state = self.state_projection(cross_scale.mean(dim=1))
            modality_embeddings = {
                modality: self._masked_mean(hidden, masks[modality])
                for modality, hidden in encoded.items()
            }
            return HumanFMOutput(
                state=state,
                scale_embeddings=scale_embeddings,
                modality_embeddings=modality_embeddings,
                available_modalities=tuple(encoded),
            )

        @staticmethod
        def _masked_mean(hidden: Tensor, valid: Tensor) -> Tensor:
            weights = valid.to(hidden.dtype).unsqueeze(-1)
            denominator = weights.sum(dim=1).clamp_min(1.0)
            return (hidden * weights).sum(dim=1) / denominator

        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())


    class PerturbationWorldModel(nn.Module):
        """Probabilistic transition model p(state[t+dt] | state[t], intervention, dt)."""

        def __init__(self, config: HumanFMConfig) -> None:
            super().__init__()
            hidden = max(config.state_dim * 2, 128)
            self.state_dim = config.state_dim
            self.intervention_dim = config.intervention_dim
            self.network = nn.Sequential(
                nn.Linear(config.state_dim + config.intervention_dim + 1, hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, config.state_dim * 2),
            )

        def forward(
            self, state: Tensor, intervention: Tensor, time_delta: Tensor
        ) -> TransitionOutput:
            if state.ndim != 2 or state.shape[1] != self.state_dim:
                raise ValueError("state has an invalid shape")
            if intervention.shape != (state.shape[0], self.intervention_dim):
                raise ValueError("intervention has an invalid shape")
            if time_delta.ndim == 1:
                time_delta = time_delta.unsqueeze(-1)
            if time_delta.shape != (state.shape[0], 1):
                raise ValueError("time_delta has an invalid shape")
            parameters = self.network(
                torch.cat(
                    [
                        state,
                        intervention.to(device=state.device, dtype=state.dtype),
                        time_delta.to(device=state.device, dtype=state.dtype),
                    ],
                    dim=-1,
                )
            )
            delta, log_variance = parameters.chunk(2, dim=-1)
            return TransitionOutput(
                next_state_mean=state + delta,
                next_state_log_variance=log_variance.clamp(-10.0, 5.0),
            )


    def contrastive_alignment_loss(
        left: Tensor,
        right: Tensor,
        valid_pairs: Tensor | None = None,
        temperature: float = 0.07,
    ) -> Tensor:
        if left.shape != right.shape or left.ndim != 2:
            raise ValueError("paired embeddings must have identical [batch, features] shapes")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if valid_pairs is not None:
            if valid_pairs.ndim != 1 or valid_pairs.shape[0] != left.shape[0]:
                raise ValueError("valid_pairs must have shape [batch]")
            governed = valid_pairs.to(device=left.device, dtype=torch.bool)
            left = left[governed]
            right = right[governed]
        if left.shape[0] < 2:
            raise ValueError("contrastive alignment requires at least two governed positive pairs")
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        logits = left @ right.transpose(0, 1) / temperature
        target = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, target) + F.cross_entropy(logits.transpose(0, 1), target)
        )


    def gaussian_transition_loss(output: TransitionOutput, target: Tensor) -> Tensor:
        if target.shape != output.next_state_mean.shape:
            raise ValueError("transition target must match the predicted state shape")
        inverse_variance = torch.exp(-output.next_state_log_variance)
        error = target - output.next_state_mean
        return 0.5 * torch.mean(
            inverse_variance * error.square() + output.next_state_log_variance
        )


else:

    class _TorchRequired:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "Human foundation-model execution requires the optional 'foundation' extra"
            ) from _TORCH_IMPORT_ERROR


    HumanFoundationModel = _TorchRequired  # type: ignore[misc,assignment]
    PerturbationWorldModel = _TorchRequired  # type: ignore[misc,assignment]

    def contrastive_alignment_loss(*_: Any, **__: Any) -> Any:
        raise RuntimeError("install the optional 'foundation' extra") from _TORCH_IMPORT_ERROR

    def gaussian_transition_loss(*_: Any, **__: Any) -> Any:
        raise RuntimeError("install the optional 'foundation' extra") from _TORCH_IMPORT_ERROR
