# Guided Decoding Workspace Setup

This repository contains setup scripts for installing and configuring SGLang and vLLM frameworks for guided decoding experiments.

## Quick Start

### Prerequisites
- Linux environment (tested on Ubuntu)
- Internet connection for downloading dependencies and models
- Sufficient disk space (~200GB+ for models and dependencies)

### Installation

1. **Build docker image:**
   ```bash
   docker build -t guided-decoding-workspace .
   ```

2. **Run the docker instance (minimum 4 GPUs required):**
   ```bash
   docker run -it guided-decoding-workspace bash
   ```

### Usage

After installation, download required model files:

```bash
# inside /workspace
sh download_models.sh
```

and launch benchmarks - PATH_TO_MODEL must be given, and PATH_TO_DATASET must direct to the dataset json file.

Without PATH_TO_DATASET, benchmark script will automatically load *squeezebits/augmented-json-mode-eval* dataset with *train* split.

```bash
# inside /workspace
# baseline (vllm)
sh benchmark_vllm_baseline.sh PATH_TO_MODEL PATH_TO_DATASET

# Optimization without quantization (sglang)
sh benchmark_sglang_bf16.sh PATH_TO_MODEL PATH_TO_DATASET

# Optimization with FP8 qunatization (for vuser 32, sglang)
sh benchmark_sglang_fp8.sh PATH_TO_MODEL PATH_TO_DATASET

# Optimization with W4A16 quantization (for vuser 8 and 16, sglang)
sh benchmark_sglang_awq.sh PATH_TO_MODEL PATH_TO_DATASET

# Optimization without qunatization, sweep (sglang)
sh benchmark_sglang_bf16_sweep.sh PATH_TO_MODEL PATH_TO_DATASET
```

Log and results can be found in *${PWD}/results/vllm* and *results/sglang* directory.

Results will also be reported to the shell after each benchmark.