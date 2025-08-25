"""
Training script for vLLM-compatible Medusa model.

This script trains Medusa heads that are compatible with vLLM's speculative decoding.
"""

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence, List
import os

import numpy as np
import torch
from torch import nn
import datasets
from datasets import Dataset, DatasetDict

import transformers
import tqdm
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from transformers.tokenization_utils import PreTrainedTokenizer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess(
    sources: List[List[Dict]],
    tokenizer: PreTrainedTokenizer,
    model_max_length: int = 2048
) -> Dict:
    """
    Preprocess conversation data for training.
    """
    # Tokenize conversations
    input_ids_list = []
    labels_list = []
    
    for conversation in tqdm.tqdm(sources):
        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            # Format conversation for chat template
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
            
            # Tokenize the full conversation
            encoding = tokenizer(
                prompt,
                max_length=model_max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            input_ids = encoding["input_ids"]
            
            # Create labels (copy of input_ids)
            labels = input_ids.copy()
            
            # Mask out non-assistant parts
            # This is a simplified version - you may need to adjust based on your template
            # Find assistant responses and only compute loss on those
            prompt_parts = prompt.split(tokenizer.eos_token)
            current_pos = 0
            
            for i, part in enumerate(prompt_parts[:-1]):  # Skip last part
                part_with_eos = part + tokenizer.eos_token
                part_tokens = tokenizer(part_with_eos, add_special_tokens=False)["input_ids"]
                
                # Check if this is an assistant response
                if "assistant" in part.lower() and i > 0:
                    # Keep these tokens for training
                    current_pos += len(part_tokens)
                else:
                    # Mask these tokens
                    for j in range(current_pos, current_pos + len(part_tokens)):
                        if j < len(labels):
                            labels[j] = IGNORE_TOKEN_ID
                    current_pos += len(part_tokens)
        else:
            # Fallback: simple concatenation
            text = ""
            labels = []
            
            for message in conversation:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "system":
                    text += f"System: {content}\n"
                elif role == "user":
                    text += f"User: {content}\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n"
            
            # Tokenize
            encoding = tokenizer(
                text,
                max_length=model_max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            input_ids = encoding["input_ids"]
            labels = input_ids.copy()  # For now, train on everything
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ids) for ids in input_ids_list],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(labs) for labs in labels_list],
        batch_first=True,
        padding_value=IGNORE_TOKEN_ID
    )
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }




def main():
    """Entry point for console script."""
    tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/guided_decoding_workspace/HF-Qwen3-32B")
    data_path = "medusa/ultrachat_data_test.json"
    model_max_length = 2048

    with open(data_path, 'r') as f:
        data = json.load(f)
    
    data_dict = preprocess(data, tokenizer, model_max_length)
    dataset = Dataset.from_dict(data_dict)
    dataset.save_to_disk("ultrachat_dataset")

if __name__ == "__main__":
    main() 