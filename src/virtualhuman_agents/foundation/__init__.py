"""Cross-scale human foundation-model framework.

The package separates data governance, representation learning, perturbation
modeling and locked evaluation. It is deliberately not a clinical decision
system and does not assert that one monolithic model can simulate a person.
"""

from .architecture import HumanFoundationModel, PerturbationWorldModel
from .curriculum import build_default_curriculum
from .model_registry import FoundationModelRegistry
from .registry import HumanDataRegistry
from .schemas import (
    BiologicalScale,
    DatasetManifest,
    HumanFMConfig,
    Modality,
    SampleIndexRecord,
)

__all__ = [
    "BiologicalScale",
    "DatasetManifest",
    "HumanDataRegistry",
    "HumanFMConfig",
    "HumanFoundationModel",
    "FoundationModelRegistry",
    "Modality",
    "PerturbationWorldModel",
    "SampleIndexRecord",
    "build_default_curriculum",
]
