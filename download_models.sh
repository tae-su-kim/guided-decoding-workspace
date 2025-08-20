#!/bin/bash

# Exit on any error
set -e
set -o pipefail

source /workspace/venvs/sglang/bin/activate

# Download model
echo "Downloading Qwen3-32B model..."
huggingface-cli download Qwen/Qwen3-32B --local-dir ./HF-Qwen3-32B
echo "Downloading Qwen3-32B-FP8 model..."
huggingface-cli download squeezebits/Qwen-32B-FP8-Dynamic --local-dir ./Qwen3-32B-FP8-Dynamic
echo "Downloading Qwen3-32B-AWQ model..."
huggingface-cli download squeezebits/Qwen3-32B-AWQ --local-dir ./Qwen3-32B-AWQ

echo "Model download completed"
