FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04

RUN apt update && apt install -y libnuma-dev libibverbs-dev python3.12 python3-pip python3-venv

RUN mkdir -p /workspace
RUN mkdir -p /workspace/results
RUN mkdir -p /workspace/venvs

# Setup sglang
COPY ./sglang-guided-decoding /workspace/sglang-guided-decoding

WORKDIR /workspace
RUN python3 -m venv venvs/sglang

WORKDIR /workspace/sglang-guided-decoding
RUN /workspace/venvs/sglang/bin/pip install -e "python[all]"
RUN /workspace/venvs/sglang/bin/pip install datasets
RUN /workspace/venvs/sglang/bin/pip install "vllm==0.9.0.1"

# Setup vllm
COPY ./vllm-guided-decoding /workspace/vllm-guided-decoding

WORKDIR /workspace
RUN python3 -m venv venvs/vllm
WORKDIR /workspace/vllm-guided-decoding

RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1 VLLM_USE_PRECOMPILED=1 /workspace/venvs/vllm/bin/pip install -e .
RUN /workspace/venvs/vllm/bin/pip install "transformers<4.54.0"
RUN /workspace/venvs/vllm/bin/pip install datasets

COPY ./download_dataset.sh /workspace/download_dataset.sh
RUN bash /workspace/download_dataset.sh

COPY ./benchmark /workspace/benchmark
COPY ./download_models.sh /workspace/download_models.sh
COPY ./benchmark_vllm_baseline.sh /workspace/benchmark_vllm_baseline.sh
COPY ./benchmark_sglang_bf16.sh /workspace/benchmark_sglang_bf16.sh
COPY ./benchmark_sglang_bf16_sweep.sh /workspace/benchmark_sglang_bf16_sweep.sh
COPY ./benchmark_sglang_fp8.sh /workspace/benchmark_sglang_fp8.sh
COPY ./benchmark_sglang_awq.sh /workspace/benchmark_sglang_awq.sh
COPY ./json_to_csv.py /workspace/json_to_csv.py
COPY ./README.md /workspace/README.md

WORKDIR /workspace
CMD sleep infinity
