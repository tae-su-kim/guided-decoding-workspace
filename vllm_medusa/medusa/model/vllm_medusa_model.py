import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig
from transformers import AutoConfig, AutoModelForCausalLM
import os
from safetensors.torch import save_file, load_file
from typing import Optional, List, Dict, Tuple
import json


class VLLMMedusaConfig(PretrainedConfig):
    """
    Configuration class for vLLM-compatible Medusa model.
    
    This config matches the format expected by vLLM's medusa implementation.
    """
    model_type = "medusa"
    
    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 32000,
        num_heads: int = 5,
        num_hidden_layers: int = 1,
        truncated_vocab_size: Optional[int] = None,
        medusa_fc_bias: bool = False,
        logit_scale: float = 1.0,
        original_lm_head: bool = False,
        architectures: List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.truncated_vocab_size = truncated_vocab_size if truncated_vocab_size else vocab_size
        self.medusa_fc_bias = medusa_fc_bias
        self.logit_scale = logit_scale
        self.original_lm_head = original_lm_head
        if architectures is None:
            self.architectures = ["MedusaModel"]
        else:
            self.architectures = architectures


class ResidualBlock(nn.Module):
    """
    Residual block for Medusa heads, matching vLLM's implementation.
    """
    def __init__(self, hidden_size: int, num_layers: int, bias: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=bias)
            for _ in range(num_layers)
        ])
        self.act = nn.SiLU()
        
        # Initialize as identity mapping
        for layer in self.layers:
            torch.nn.init.zeros_(layer.weight)
            if bias:
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x


class VLLMMedusaModel(nn.Module,PyTorchModelHubMixin):
    """
    vLLM-compatible Medusa model implementation.
    
    This model only contains the Medusa heads and is designed to be compatible
    with vLLM's speculative decoding framework.
    """
    
    def __init__(self, config: VLLMMedusaConfig, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_heads
        self.truncated_vocab_size = config.truncated_vocab_size
        
        # Create residual blocks for each head
        self.blocks = nn.ModuleList([
            ResidualBlock(
                hidden_size=config.hidden_size,
                num_layers=config.num_hidden_layers,
                bias=config.medusa_fc_bias
            ).to(torch_dtype)
            for _ in range(config.num_heads)
        ])
        
        # Create LM heads
        if config.original_lm_head:
            # Share the first LM head across all heads
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(torch_dtype)
            self.lm_heads = nn.ModuleList([self.lm_head for _ in range(config.num_heads)])
        else:
            # Separate LM head for each Medusa head
            self.lm_heads = nn.ModuleList([
                nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(torch_dtype)
                for _ in range(config.num_heads)
            ])
        
        # Optional token map for vocabulary truncation
        self.token_map = None
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all Medusa heads.
        
        Args:
            hidden_states: Hidden states from the base model [batch_size, seq_len, hidden_size]
            
        Returns:
            List of logits from each Medusa head
        """
        outputs = []
        for block, lm_head in zip(self.blocks, self.lm_heads):
            # Pass through residual block
            head_hidden = block(hidden_states)
            # Get logits from LM head
            logits = lm_head(head_hidden)
            outputs.append(logits)
        
        return outputs
    
    @classmethod
    def from_base_model(
        cls,
        base_model_name_or_path: str,
        num_heads: int = 5,
        num_hidden_layers: int = 1,
        **kwargs
    ):
        """
        Create a Medusa model from a base model configuration.
        """
        # Load base model config
        base_config = AutoConfig.from_pretrained(base_model_name_or_path)
        
        # Create Medusa config
        medusa_config = VLLMMedusaConfig(
            hidden_size=base_config.hidden_size,
            vocab_size=base_config.vocab_size,
            num_heads=num_heads,
            num_hidden_layers=num_hidden_layers,
            **kwargs
        )
        
        # Create and return model
        return cls(medusa_config)
    
    # def save_pretrained(self, save_directory: str):
    #     """
    #     Save the model in vLLM-compatible format.
    #     """
    #     os.makedirs(save_directory, exist_ok=True)
        
    #     # Save config
    #     self.config.save_pretrained(save_directory)
        
    #     # Prepare state dict for vLLM format
    #     state_dict = self.state_dict()
        
    #     # Save token map if exists
    #     if self.token_map is not None:
    #         state_dict["token_map"] = self.token_map
        
    #     # Save as safetensors
    #     save_file(state_dict, os.path.join(save_directory, "model.safetensors"))
    
    # @classmethod
    # def from_pretrained(cls, model_path: str, torch_dtype: torch.dtype = torch.bfloat16):
    #     """
    #     Load a vLLM-compatible Medusa model.
    #     """
    #     # Load config
    #     config = VLLMMedusaConfig.from_pretrained(model_path)
        
    #     # Create model
    #     model = cls(config)
        
    #     # Load weights
    #     state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    #     model.load_state_dict(state_dict)
        
        
    #     return model


class MedusaModelWithBase(nn.Module):
    """
    Medusa model that includes the base model for training.
    This is used during training to have both base model and Medusa heads together.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        medusa_config: VLLMMedusaConfig,
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.base_model = base_model
        self.medusa = VLLMMedusaModel(medusa_config, torch_dtype=torch_dtype)
        self.config = medusa_config
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass through base model and Medusa heads.
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get hidden states from last layer
        hidden_states = outputs.hidden_states[-1]
        
        # Get Medusa logits
        medusa_logits = self.medusa(hidden_states)
        
        # Stack logits for loss computation
        # Shape: [num_heads, batch_size, seq_len, vocab_size]
        medusa_logits = torch.stack(medusa_logits, dim=0)
        
        return {
            'logits': medusa_logits,
            'base_logits': outputs.logits if hasattr(outputs, 'logits') else None,
            'hidden_states': hidden_states,
            'labels': labels
        }
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model in vLLM-compatible format.
        """
        self.medusa.save_pretrained(save_directory)
        self.medusa.config.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path: str,
        medusa_heads_path: Optional[str] = None,
        num_heads: int = 5,
        num_hidden_layers: int = 1,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        """
        Load a model with base model and Medusa heads.
        """
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            **kwargs
        )
        
        # Create or load Medusa config
        if medusa_heads_path and os.path.exists(os.path.join(medusa_heads_path, "config.json")):
            medusa_config = VLLMMedusaConfig.from_pretrained(medusa_heads_path)
        else:
            base_config = base_model.config
            medusa_config = VLLMMedusaConfig(
                hidden_size=base_config.hidden_size,
                vocab_size=base_config.vocab_size,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers
            )
        
        # Create model
        model = cls(base_model, medusa_config, torch_dtype=torch_dtype)
        
        # Load Medusa heads if path provided
        if medusa_heads_path and os.path.exists(os.path.join(medusa_heads_path, "model.safetensors")):
            model.medusa = VLLMMedusaModel.from_pretrained(medusa_heads_path, torch_dtype=torch_dtype)
        
        return model 