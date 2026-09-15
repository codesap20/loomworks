FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=8

RUN mkdir -p /app/checkpoints /cache /workspace
WORKDIR /workspace

# ---- system stack: standard-arch tasks (instruct / dpo / grpo) --------------
RUN pip install -U pip packaging==23.2 setuptools==75.8.0 wheel ninja uv && \
    uv pip install --system \
        transformers==4.51.3 trl==0.18.0 peft==0.15.1 accelerate==1.6.0 \
        deepspeed==0.15.4 bitsandbytes==0.45.4 datasets==3.5.0 \
        liger-kernel==0.5.9 sentencepiece tiktoken==0.9.0 tenacity==9.1.2 \
        textstat==0.7.8 && \
    (pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
     || pip install flash-attn==2.7.4.post1 --no-build-isolation) && \
    uv pip install --system vllm==0.8.3 && \
    rm -rf /root/.cache/uv /root/.cache/pip

# ---- axolotl venv: tokenization only (formatting parity with the validator) --
RUN python -m venv /opt/axo && \
    /opt/axo/bin/pip install -U pip packaging==23.2 setuptools==75.8.0 wheel ninja uv && \
    /opt/axo/bin/uv pip install --python /opt/axo/bin/python --no-build-isolation axolotl==0.9.1 && \
    rm -rf /root/.cache/uv /root/.cache/pip

# ---- modern venv: transformers v5 for custom hybrid-linear-attention archs ---
# quasar_text / quasar_long remote code hard-imports fla + causal_conv1d and
# uses transformers.initialization (v5-only), so this cannot share the system stack.
RUN python -m venv /opt/modern && \
    /opt/modern/bin/pip install -U pip wheel ninja packaging setuptools && \
    /opt/modern/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 && \
    /opt/modern/bin/pip install \
        "transformers==5.9.0" "accelerate>=1.8" "datasets>=3.5" \
        safetensors sentencepiece einops triton && \
    /opt/modern/bin/pip install flash-linear-attention && \
    (/opt/modern/bin/pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
     || /opt/modern/bin/pip install causal-conv1d --no-build-isolation) && \
    (/opt/modern/bin/pip install cut-cross-entropy || true) && \
    rm -rf /root/.cache/uv /root/.cache/pip

COPY src /workspace/src
COPY ds_config /workspace/ds_config
COPY ops/accelerate /workspace/ops/accelerate
COPY ops/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
