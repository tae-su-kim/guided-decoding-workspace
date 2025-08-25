"""
vLLM-compatible Medusa model implementations.
"""

from .vllm_medusa_model import (
    VLLMMedusaConfig,
    VLLMMedusaModel,
    MedusaModelWithBase,
    ResidualBlock
)

__all__ = [
    "VLLMMedusaConfig",
    "VLLMMedusaModel",
    "MedusaModelWithBase", 
    "ResidualBlock"
] 