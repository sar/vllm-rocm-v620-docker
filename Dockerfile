# syntax=docker/dockerfile:1.4
# =============================================================================
# vLLM v0.20.0 on AMD gfx1030 (Radeon Pro V620) — ROCm 7.2.1
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
    HIP_VISIBLE_DEVICES=all \
    PYTORCH_ROCM_ARCH=gfx1030 \
    GPU_TARGETS=gfx1030 \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

WORKDIR /workspace

# =============================================================================
# Stage 1: Install uv + system build dependencies
# =============================================================================
FROM base AS builder

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ONLY install what the base image doesn't have. 
# DO NOT reinstall rocm-hip-sdk or rocm-device-libs; the base image already has them.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Create venv + install deps + compile vLLM
# =============================================================================
FROM builder AS vllm-build

RUN uv venv /opt/venv --system-site-packages
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Pre-install Python dependencies using uv (fast)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
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

# Build vLLM. 
# CRITICAL: Using `pip` instead of `uv` here. If the C++ build fails, `pip` 
# immediately fails with the real compiler error. `uv` falls back to PyPI, 
# grabs the CUDA wheel, and masks the error with httplib exceptions.
# --no-build-isolation uses the base image's PyTorch ROCm headers.
RUN --mount=type=cache,target=/root/.cache/pip \
    GPU_TARGETS=gfx1030 \
    PYTORCH_ROCM_ARCH=gfx1030 \
    BUILD_FA=0 \
    VLLM_FLASH_ATTN=0 \
    VLLM_FLASHINFER=0 \
    MAX_JOBS=$(nproc) \
    pip install --no-cache-dir --no-build-isolation -v . 2>&1 | tee /tmp/vllm_build.log

# Verify build succeeded
RUN python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed')" || { \
    echo "❌ Build failed. Compiler errors:"; \
    grep -i "error\|fatal\|hipcc: error" /tmp/vllm_build.log | tail -30; \
    exit 1; }

# =============================================================================
# Stage 3: Final minimal inference image
# =============================================================================
FROM base AS final

# Copy the built venv
COPY --from=vllm-build /opt/venv /opt/venv

# Set runtime environment
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HIP_VISIBLE_DEVICES=all \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0

# Create necessary directories as root
RUN mkdir -p /workspace/models /workspace/.cache/huggingface

EXPOSE 8000

# Default command
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]