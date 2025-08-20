#!/bin/bash

# Exit on any error
set -e
set -o pipefail

source /workspace/venvs/sglang/bin/activate

# Download model
echo "Downloading dataset..."
huggingface-cli download squeezebits/augmented-json-mode-eval --repo-type dataset

echo "Model download completed"
