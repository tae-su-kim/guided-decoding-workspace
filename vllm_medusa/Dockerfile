FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel


WORKDIR /workspace

COPY . /workspace/medusa_workspace


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git git-lfs ca-certificates build-essential pkg-config cmake wget curl

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r /workspace/medusa_workspace/requirements.txt

RUN pip install -e /workspace/medusa_workspace


RUN git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered /workspace/medusa_workspace/data/ShareGPT_Vicuna_unfiltered
RUN huggingface-cli download --repo-type dataset HuggingFaceH4/ultrachat_200k --local-dir /workspace/medusa_workspace/data/ultrachat_200k

CMD sleep infinity