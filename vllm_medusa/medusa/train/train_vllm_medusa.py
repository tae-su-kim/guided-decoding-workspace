"""
Training script for vLLM-compatible Medusa model.

This script trains Medusa heads that are compatible with vLLM's speculative decoding.
"""

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence, List, Literal
import os
import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import random
import transformers
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.hf_argparser import HfArgumentParser
from safetensors.torch import save_file

from torch.nn import CrossEntropyLoss
import deepspeed

# Import our vLLM-compatible model
from medusa.model.vllm_medusa_model import MedusaModelWithBase

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class MedusaTrainer(Trainer):
    """
    Custom trainer for Medusa model that handles the special loss computation.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss for Medusa heads.
        
        The loss is computed for each Medusa head predicting future tokens.
        """
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        
        logits = outputs['logits']  # [num_heads, batch_size, seq_len, vocab_size]
        labels = inputs["labels"]  # [batch_size, seq_len]

        # Get dimensions
        num_heads = logits.shape[0]
        batch_size = logits.shape[1]
        seq_len = logits.shape[2]
        vocab_size = logits.shape[3]
        
        # Compute loss for each head
        loss = 0
        loss_fct = CrossEntropyLoss(ignore_index=IGNORE_TOKEN_ID)
        
        # Dictionary to store metrics
        log = {}
        
        for i in range(num_heads):
            # Each head predicts (i+1) tokens ahead
            # Shift logits and labels appropriately
            if seq_len > (2 + i):
                # Get logits for this head (excluding last 2+i positions)
                head_logits = logits[i, :, :-(2+i), :].contiguous()
                # Get corresponding labels (shifted by 2+i positions)
                head_labels = labels[:, 2+i:].contiguous()
                
                # Reshape for loss computation
                head_logits = head_logits.view(-1, vocab_size)
                head_labels = head_labels.view(-1)
                
                # Compute loss for this head
                loss_i = loss_fct(head_logits, head_labels)
                loss += loss_i
                
                # Compute accuracy metrics
                with torch.no_grad():
                    # Get valid (non-ignored) labels
                    not_ignore = head_labels.ne(IGNORE_TOKEN_ID)
                    valid_labels = head_labels[not_ignore]
                    valid_logits = head_logits[not_ignore]
                    
                    if valid_labels.numel() > 0:
                        # Top-1 accuracy
                        predictions = valid_logits.argmax(dim=-1)
                        correct = predictions.eq(valid_labels)
                        accuracy = correct.float().mean().item()
                        log[f"medusa{i}_accuracy"] = accuracy
                        
                        # Top-5 accuracy
                        _, top5 = valid_logits.topk(5, dim=-1)
                        correct_top5 = top5.eq(valid_labels.unsqueeze(-1)).any(-1)
                        accuracy_top5 = correct_top5.float().mean().item()
                        log[f"medusa{i}_top5_accuracy"] = accuracy_top5
                
                log[f"medusa{i}_loss"] = loss_i.item()
        
        # Average loss across heads
        loss = loss / num_heads
        
        # Log metrics
        self.log(log)

        return (loss, outputs) if return_outputs else loss


    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save only Medusa heads to avoid shared tensor issues.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        if output_dir is None:
            raise ValueError("output_dir cannot be None")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        assert(isinstance(self.model, MedusaModelWithBase))

        self.model.config.save_pretrained(output_dir)

        # self.accelerator.get_state_dict(self.deepspeed)
        with deepspeed.zero.GatheredParameters(self.model.medusa.parameters()):
            save_file(self.model.medusa.state_dict(), os.path.join(output_dir, "medusa_heads.safetensors"))

        training_args_path = os.path.join(output_dir, "training_args.bin")
        torch.save(self.args, training_args_path)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="lmsys/vicuna-7b-v1.3",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit mode"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit mode"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation data"}
    )
    lazy_preprocess: bool = field(
        default=True,
        metadata={"help": "Whether to use lazy preprocessing"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    # Medusa-specific arguments
    medusa_num_heads: int = field(
        default=5,
        metadata={"help": "Number of Medusa heads"}
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head"}
    )
    medusa_fc_bias: bool = field(
        default=False,
        metadata={"help": "Whether to use bias in Medusa FC layers"}
    )
    original_lm_head: bool = field(
        default=False,
        metadata={"help": "Whether to share the same LM head across all Medusa heads"}
    )

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    # Apply prompt templates
    conversations = []
    prompts = []
    # # import pdb; pdb.set_trace()
    for i, conversation in enumerate(sources):
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        prompts.append(prompt)
        conversations.append(conversation)

    # Tokenize conversations
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets. Only compute loss on the assistant outputs.
    for conv_index, (conversation, target, prompt) in enumerate(zip(conversations, targets, prompts)):
        for idx, turn in enumerate(conversation):
            if turn["role"] == "assistant":
                try:
                    if "reasoning_content" in turn and isinstance(turn["reasoning_content"], str) and len(turn["reasoning_content"]) > 0:
                        content = turn["reasoning_content"]
                    else:
                        content = turn["content"]
                    # Unfortunate strip() necessary because chat templates are doing the same.
                    start = prompt.index(content.strip())
                    stop = start + len(content)
                except:
                    # if turn is longer than max length, skip
                    # print(f"skippiqng turn({idx}) {content[:10]} not in prompt{prompt[:10]}")
                    continue
                indices= []
                for tok_index, (tok_start, tok_stop) in enumerate(encoding.offset_mapping[conv_index]):
                    if tok_stop >= start and tok_start < stop:
                        indices.append(tok_index)
                target[indices] = encoding.input_ids[conv_index][indices]
                    

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, raw_data: List, tokenizer: PreTrainedTokenizer, model_max_length: int = 2048):
        super().__init__()
        
        print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer,)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i]
        }

class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        random.shuffle(self.raw_data)
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    training_args: TrainingArguments
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    # Load training data
    print("Loading training data...")
    with open(data_args.data_path, 'r') as f:
        train_json = json.load(f)
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    train_dataset = dataset_cls(
        train_json,
        tokenizer=tokenizer
    )
    
    # Load eval data if provided
    eval_dataset = None
    if data_args.eval_data_path:
        print("Loading evaluation data...")
        with open(data_args.eval_data_path, 'r') as f:
            eval_json = json.load(f)
        eval_dataset = SupervisedDataset(
            eval_json,
            tokenizer=tokenizer,
            model_max_length=training_args.model_max_length
        )
    
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with Medusa heads
    print("Loading model...")
    model = MedusaModelWithBase.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        num_heads=training_args.medusa_num_heads,
        num_hidden_layers=training_args.medusa_num_layers,
        load_in_8bit=model_args.load_in_8bit,
        load_in_4bit=model_args.load_in_4bit,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
    )
    
    # Update config with additional parameters
    model.config.medusa_fc_bias = training_args.medusa_fc_bias
    model.config.original_lm_head = training_args.original_lm_head
    
    # Prepare data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args
    )
    
    # Create output directory name
    output_dir = training_args.output_dir
    if not output_dir.endswith("_vllm_medusa"):
        output_dir = f"{output_dir}_vllm_medusa_{model_args.model_name_or_path.split('/')[-1]}_heads{training_args.medusa_num_heads}_layers{training_args.medusa_num_layers}"
        training_args.output_dir = output_dir
    
    # Initialize trainer
    trainer = MedusaTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    trainer.local_rank = int(os.environ["LOCAL_RANK"])

    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save only Medusa heads in vLLM format (avoid base model with shared tensors)
    print("Saving vLLM-compatible Medusa heads...")
    vllm_output_dir = os.path.join(training_args.output_dir, "vllm_medusa_heads")
    
    # Custom save that only saves Medusa heads
    os.makedirs(vllm_output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(vllm_output_dir)
    with deepspeed.zero.GatheredParameters(model.medusa.parameters()):
        # Save Medusa heads using model's method
        model.medusa.save_pretrained(vllm_output_dir)
    
    # Save training state for reference
    trainer.save_state()
    
    print(f"Training complete! vLLM-compatible Medusa heads saved to: {vllm_output_dir}")
    print(f"You can use these heads with vLLM by specifying --speculative-model {vllm_output_dir}")


def main():
    """Entry point for console script."""
    train()

if __name__ == "__main__":
    main() 