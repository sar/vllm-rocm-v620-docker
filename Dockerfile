# syntax=docker/dockerfile:1.4
# =============================================================================
# Dockerfile: vLLM v0.20.0 Inference + AMD gfx1030 + ROCm 7.2.1
# =============================================================================

ARG BASE_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE} AS base

# =============================================================================
# Environment Configuration
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # ROCm + gfx1030
    HIP_VISIBLE_DEVICES=all \
    PYTORCH_ROCM_ARCH=gfx1030 \
    GPU_TARGETS=gfx1030 \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    BNB_ROCM_ARCH=gfx1030 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Hugging Face cache
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

WORKDIR /workspace

# =============================================================================
# Stage 1: Install uv + system build dependencies
# =============================================================================
FROM base AS builder

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git rocm-hip-sdk rocm-device-libs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Create venv + install deps + compile vLLM
# =============================================================================
FROM builder AS vllm-build

# Create isolated venv (inherits base ROCm PyTorch via --system-site-packages for stability)
RUN uv venv /opt/venv --system-site-packages
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Install verified compatible stack for vLLM v0.20.x + ROCm 7.2.1
RUN uv pip install --no-cache \
    "transformers>=4.53.0" \
    "ray[default]>=2.40.0" \
    "pydantic>=2.10.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "aiohttp>=3.11.0" \
    "tiktoken>=0.9.0" \
    "sentencepiece>=0.2.0" \
    "prometheus-client>=0.21.0" \
    "huggingface-hub>=0.28.0" \
    "tokenizers>=0.21.0" \
    "numpy>=1.26,<2.1" \
    "xgrammar>=0.1.10" \
    "lm-format-enforcer>=0.10.0"

# Clone vLLM v0.20.0
ARG VLLM_VERSION=v0.20.0
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /tmp/vllm-src
WORKDIR /tmp/vllm-src

# Modern vLLM reads GPU_TARGETS directly from env during build.
# No fragile sed patches needed.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy \
    BUILD_FA=0 \
    PYTORCH_ROCM_ARCH=gfx1030 \
    GPU_TARGETS=gfx1030 \
    MAX_JOBS=$(nproc) \
    uv pip install --no-cache . --verbose 2>&1 | tee /tmp/vllm_build.log

# Verify
RUN python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed')" && \
    python -c "from vllm import _custom_ops; print('✅ ROCm custom ops loaded')"

# =============================================================================
# Stage 3: Final minimal inference image
# =============================================================================
FROM base AS final

# Copy entire isolated venv
COPY --from=vllm-build /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Create non-root user
RUN useradd -m -u 1000 -G video,render vllmuser && \
    mkdir -p /workspace/models /workspace/.cache/huggingface /workspace/scripts && \
    chown -R vllmuser:vllmuser /workspace

USER vllmuser
EXPOSE 8000

# Health check
COPY <<'EOF' /workspace/scripts/health_check.sh
#!/bin/bash
set -e
echo "🔍 Checking ROCm + vLLM environment..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__} (ROCm: {torch.version.hip})')"
python -c "import vllm; print(f'✅ vLLM {vllm.__version__} ready')"
python -c "from vllm import _custom_ops; print('✅ Custom ops loaded')"
echo "✅ Health check passed"
EOF
RUN chmod +x /workspace/scripts/health_check.sh

CMD ["/workspace/scripts/health_check.sh"]