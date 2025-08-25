"""
vLLM Medusa: Speculative Decoding with Multiple Decoding Heads

This package provides vLLM-compatible Medusa model implementation and training code
for accelerated language model inference through speculative decoding.

Medusa speeds up LLM inference by using multiple decoding heads to predict several
future tokens in parallel, reducing the number of forward passes needed.
"""

__version__ = "1.0.0"
__author__ = "vLLM Medusa Team"
__email__ = "medusa@vllm.ai"

from .medusa.model.vllm_medusa_model import (
    VLLMMedusaConfig,
    VLLMMedusaModel,
    MedusaModelWithBase,
    ResidualBlock
)

from .medusa.train.train_vllm_medusa import (
    ModelArguments,
    DataArguments, 
    TrainingArguments,
    MedusaTrainer
)

__all__ = [
    "VLLMMedusaConfig",
    "VLLMMedusaModel", 
    "MedusaModelWithBase",
    "ResidualBlock",
    "ModelArguments",
    "DataArguments",
    "TrainingArguments", 
    "MedusaTrainer"
] 