# vLLM Medusa

This is a Medusa training code that references [FasterDecoding/Medusa's training code](https://github.com/FasterDecoding/Medusa?tab=readme-ov-file#training-on-various-architectures) to create modules compatible with the latest DeepSpeed and usable in vLLM format.


## 🎯 Complete Workflow: UltraChat & Shared GPT → vLLM Medusa

### 0. install

pip install -e .

### 1. Prepare Training Data

```bash
#  serve original llm
vllm serve Qwen/Qwen3-32B\
    --dtype bfloat16 \
    --host 127.0.0.1 \
    --port 30000 \
    --gpu-memory-utilization 0.95 \
    --tokenizer Qwen/Qwen3-32B \
    --tensor-parallel-size 2 \
    --max-num-seqs 256 \
    --max-model-len 2048 \
    --tool-call-parser hermes \
    --disable-log-requests \
    --reasoning-parser qwen3

```


```bash
# Load UltraChat 200k and convert for Medusa training
# huggingface-cli download --repo-type dataset HuggingFaceH4/ultrachat_200k
python -m medusa.train.create_data_ultrachat \
    --dataset-name data/ultrachat_200k \
    --split train_sft \
    --output-filename ultrachat_data.json \
    --model-name Qwen/Qwen3-32B
```


```bash
# git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered
python -m medusa.train.create_data_share_gpt \
    --input-filename data/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --output-filename ShareGPT_data_cleaned_split.json \
    --model-name Qwen/Qwen3-32B

python -m medusa.train.create_data_share_gpt \
    --input-filename data/ShareGPT_Vicuna_unfiltered/ShareGPT_2023.05.04v0_Wasteland_Edition.json \
    --output-filename ShareGPT_data_Wasteland_Edition.json \
    --model-name Qwen/Qwen3-32B

```

### 2. Train Medusa Heads

**Notes:**
> - Before starting training, you may need to combine the individual data JSON files you generated into one.
> - DeepSpeed state files are not saved, so resume functionality is not properly supported.
> - To use the `deepspeed.json` file, you may need to change the type hint in `transformers.TrainingArguments` as follows: <br> `deepspeed: Optional[Union[dict, str]]` -> `deepspeed: Optional[str]`

```bash
# Train Medusa heads
python -m medusa.train.train_vllm_medusa \
    --model_name_or_path Qwen/Qwen3-32B \
    --data_path medusa_training_data.json \
    --output_dir ./medusa_heads \
    --medusa_num_heads 3 \
    --medusa_num_layers 1 \
    --num_train_epochs 2
```
or

```bash
torchrun --nproc_per_node=3 medusa/train/train_vllm_medusa.py \
    --model_name_or_path Qwen/Qwen3-32B \
    --data_path  combined_data.json \
    --bf16 True \
    --output_dir trainig_output \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save-only-model True \
    --save_steps 200 \
    --save_total_limit 50 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 3 \
    --medusa_num_layers 1 \
    --deepspeed deepspeed.json 2>&1 | tee training_log.log
```

### 3. Serve with vLLM
```bash
# Start vLLM server with Medusa speculative decoding
vllm serve Qwen/Qwen3-32B \
    --speculative-config '{
        "method": "medusa", 
        "model": "./medusa_heads/vllm_medusa_heads", 
        "num_speculative_tokens": 3
    }'
```
